import os
import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/goodreads'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def generate_split():
    # Define the size of each split (you can adjust these ratios as needed)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Load the preprocessed data
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}

    # Total number of books
    total_books = len(book_dict)
    
    # Shuffle the indices of the books
    book_indices = np.random.permutation(total_books)
    
    # Calculate the number of books in each split
    num_train = int(total_books * train_ratio)
    num_val = int(total_books * val_ratio)
    
    # Indices for train, validation, and test splits
    train_indices = book_indices[:num_train]
    val_indices = book_indices[num_train:num_train+num_val]
    test_indices = book_indices[num_train+num_val:]
    
    # Create a folder for the split
    os.makedirs(f'{path}/split', exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))

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

        # Authors
        for author in book_data.get('authors', []):
            author_id = author.get('author_id', '')
            if isinstance(author_id, (int, str)): #check if author_id is hashable
                if author_id not in nodes:
                    nodes[author_id] = len(nodes)
            else:
                print(f"Ignoring author with non-hashable author_id: {author_id}")

        for author_id in book_data.get('authors', []):
            author_id = author.get('author_id', '')
            if isinstance(author_id, (int, str)): #check if author_id is hashable
                if author_id not in nodes:
                    edges.append({'src': nodes[author_id], 'edge_attr': 'author', 'dst': book_id})
            else:
                print(f"Ignoring author with non-hashable author_id: {author_id}")
                
        # Similar books
        for similar_book in book_data.get('similar_books', []):
            if similar_book not in nodes:
                nodes[similar_book] = len(nodes)
            edges.append({'src': book_id, 'edge_attr': 'similar_book', 'dst': nodes[similar_book]})

        # Shelves
        for shelf in book_data.get('popular_shelves', []):
            shelf_name = shelf.get('name', '')
            if shelf_name not in nodes:
                nodes[shelf_name] = len(nodes)
            edges.append({'src': book_id, 'edge_attr': 'shelf', 'dst': nodes[shelf_name]})

        # Format
        book_format = book_data.get('format', '')
        if book_format not in nodes:
            nodes[book_format] = len(nodes)
        edges.append({'src': book_id, 'edge_attr': 'format', 'dst': nodes[book_format]})

        # Publisher
        publisher = book_data.get('publisher', '')
        if publisher not in nodes:
            nodes[publisher] = len(nodes)
        edges.append({'src': book_id, 'edge_attr': 'publisher', 'dst': nodes[publisher]})

        # Language code
        language_code = book_data.get('language_code', '')
        if language_code not in nodes:
            nodes[language_code] = len(nodes)
        edges.append({'src': book_id, 'edge_attr': 'language_code', 'dst': nodes[language_code]})

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
    #print('Encoding book descriptions...')
    #descriptions = [book['title'] + " " + book['description'] for book in book_dict.values()]
    text2embedding = load_text2embedding[model_name]
    
    # Initialize a tqdm progress bar
    #pbar = tqdm(total=len(descriptions))

    #description_embeddings = []
    #for description in descriptions:
     #    Encode each description and append to the list
      #  embedding = text2embedding(model, tokenizer, device, [description])
       # description_embeddings.append(embedding)
        #pbar.update(1)  # Update the progress bar
    #pbar.close()  # Close the progress bar
    
    #print("Number of description embeddings:", len(description_embeddings))

    # Combine features for each book
    #book_features = {}
    #for book_id, desc_vec in zip(book_dict.keys(), description_embeddings):
     #   book_features[book_id] = desc_vec.numpy()
    #print("Number of books with features:", len(book_features))

    # Stack the flattened feature vectors to create the feature matrix
    #book2feature = np.vstack(list(book_features.values()))
    #book2list = list(book_dict.keys())

    # Check shapes
    #print("Number of books:", len(book_features))
    #print("Feature vector shape:", book2feature.shape)
    
    # Save description embeddings
    #d_embs = torch.tensor(book2feature, dtype=torch.float)
    #torch.save(d_embs, f'{path}/d_embs.pt')
    
    # Encode graphs
    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for book_id in tqdm(book_dict.keys()):
        # Load nodes and edges
        nodes = pd.read_csv(f'{path_nodes}/{book_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{book_id}.csv')

        # Node features
        #node_features = text2embedding(model, tokenizer, device, nodes['node_attr'].tolist()).numpy()
        nodes.node_attr.fillna("", inplace=True)
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())

        # Edges
        #edge_features = text2embedding(model, tokenizer, device, edges['edge_attr'].tolist()).numpy()
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges['src'].tolist(), edges['dst'].tolist()])
        
        

        # Create PyG graph data object
        graph_data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,num_nodes=len(nodes))

        # Save graph data object
        torch.save(graph_data, f'{path_graphs}/{book_id}.pt')

if __name__ == '__main__':    
    #step_one()
    step_two()
    generate_split()