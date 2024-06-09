import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/goodreads'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


def generate_split():
    # Load the preprocessed data
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}

    # Create a list of book IDs
    book2list = list(book_dict.keys())

    # Create labels for each book based on its genres
    labels = []
    for index, book in enumerate(book2list):
        labels.append([index, book_dict[book]['genres']])

    # Extract unique genres and create a mapping
    label_genres = [label[1] for label in labels]
    dict_from_set = {element: index for index, element in enumerate(set(label_genres))}

    # Map each book's genre to its corresponding index
    labels_v2 = []
    for index, book in enumerate(book2list):
        labels_v2.append([index, dict_from_set[book_dict[book]['genres']]])

    # Define fixed sizes for the splits (assuming the split size used by the other person)
    num_train = 800
    num_val = 400
    num_test = 2000

    # Create train, validation, and test splits
    labels_train = labels_v2[:num_train]
    labels_val = labels_v2[num_train:num_train+num_val]
    labels_test = labels_v2[num_train+num_val:num_train+num_val+num_test]
    labels_final = [labels_train, labels_val, labels_test]

    # Extract book IDs for the test set
    test_book_ids = book2list[num_train+num_val:num_train+num_val+num_test]

    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_train])))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_val])))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_test])))

def step_one():
    # Load the preprocessed data
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))

    # Construct book_dict: key book_id, value book_info_dict
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}

    # Generate nodes and edges
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    for book_id, book_data in tqdm(book_dict.items()):
        nodes = {}
        edges = []

        #Added new to add description(till authors)
        def add_node(node_attr, node_dict, node_list):
            if node_attr not in node_dict:
                node_dict[node_attr] = len(node_dict)
            return node_dict[node_attr]

        def add_edge(src, edge_attr, dst, edge_list):
            edge_list.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

        # Book Node (Add description as node feature)
        book_node_id = add_node(book_id, nodes, nodes)

        # Description node
        description = book_data.get('description', '')
        if description:
            desc_node = add_node(description, nodes, nodes)
            add_edge(book_node_id, 'description', desc_node, edges)

        # Authors
        for author in book_data.get('authors', []):
            author_id = author.get('author_id', '')
            if isinstance(author_id, (int, str)):
                if author_id not in nodes:
                    nodes[author_id] = len(nodes)

        for author_id in book_data.get('authors', []):
            author_id = author.get('author_id', '')
            if isinstance(author_id, (int, str)): #check if author_id is hashable
                author_node = add_node(author_id, nodes, nodes)
                add_edge(book_node_id, 'author', author_node, edges)
            else:
                print(f"Ignoring author with non-hashable author_id: {author_id}")
                
        # Similar books
        with open ('dataset/goodreads/goodreads_books.json') as f:
            all_books_data = json.load(f) 

        for similar_book in book_data.get('similar_books', []):
            #Check if the book exists in the larger dataset
            if similar_book in all_books_data:
                similar_book_node = add_node(similar_book, nodes, nodes)
                add_edge(book_node_id, 'similar_book', similar_book_node, edges)
            else:
                print(f"Book ID {similar_book} not found in the dataset.")

        # Shelves
        #for shelf in book_data.get('popular_shelves', []):
         #   shelf_name = shelf.get('name', '')
          #  shelf_node = add_node(shelf_name, nodes, nodes)
           # add_edge(book_node_id, 'shelf', shelf_node, edges)

        # Format
        book_format = book_data.get('format', '')
        format_node = add_node(book_format, nodes, nodes)
        add_edge(book_node_id, 'format', format_node, edges)

        # Publisher
        publisher = book_data.get('publisher', '')
        publisher_node = add_node(publisher, nodes, nodes)
        add_edge(book_node_id, 'publisher', publisher_node, edges)

        # Language code
        language_code = book_data.get('language_code', '')
        language_node = add_node(language_code, nodes, nodes)
        add_edge(book_node_id, 'language_code', language_node, edges)

        # Save nodes and edges to CSV files
        nodes_df = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
        edges_df = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes_df.to_csv(f'{path_nodes}/{book_id}.csv', index=False)
        edges_df.to_csv(f'{path_edges}/{book_id}.csv', index=False)

def step_two():
    # Load the preprocessed data
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}

    # Load the model and tokenizer
    model, tokenizer, device = load_model[model_name]()

    # Encode book descriptions
    print('Encoding book descriptions...')
    descriptions = [book['title'] + " " + book['description'] for book in book_dict.values()]
    text2embedding = load_text2embedding[model_name]
    
    # Initialize a tqdm progress bar
    pbar = tqdm(total=len(descriptions))

    description_embeddings = []
    for description in descriptions:
      #  Encode each description and append to the list
        embedding = text2embedding(model, tokenizer, device, [description])
        description_embeddings.append(embedding)
        pbar.update(1)  # Update the progress bar
    pbar.close()  # Close the progress bar
    
    print("Number of description embeddings:", len(description_embeddings))

    # Combine features for each book
    book_features = {}
    for book_id, desc_vec in zip(book_dict.keys(), description_embeddings):
        book_features[book_id] = desc_vec.numpy()
    print("Number of books with features:", len(book_features))

    # Stack the flattened feature vectors to create the feature matrix
    book2feature = np.vstack(list(book_features.values()))
    book2list = list(book_dict.keys())

    # Check shapes
    print("Number of books:", len(book_features))
    print("Feature vector shape:", book2feature.shape)
    
    # Save description embeddings
    d_embs = torch.tensor(book2feature, dtype=torch.float)
    torch.save(d_embs, f'{path}/d_embs.pt')
    
    # Encode graphs
    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for book_id in tqdm(book_dict.keys()):
        # Load nodes and edges
        nodes = pd.read_csv(f'{path_nodes}/{book_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{book_id}.csv')

        # Node features
        nodes.node_attr.fillna("", inplace=True)
    
        # Concatenate along feature dimension
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # Edges
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges['src'].tolist(), edges['dst'].tolist()])
        
        # Create PyG graph data object
        graph_data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,num_nodes=len(nodes))

        # Save graph data object
        torch.save(graph_data, f'{path_graphs}/{book_id}.pt')

def generate_question_embeddings():
    # Load the preprocessed data
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}

    # Load the model and tokenizer
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # Create default questions for each book
    questions = [f"Predict the genre (fiction, non-fiction or romance) of the book '{book['title']}' based on the description." for book in book_dict.values()]

    # Encode questions
    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

if __name__ == '__main__':    
    step_one()
    generate_question_embeddings()
    step_two()
    generate_split()
