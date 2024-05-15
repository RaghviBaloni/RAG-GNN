import os
import pandas as pd
import numpy as np
import json
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils.lm_modeling import load_model, load_text2embedding

# Define the model loading function
#def load_model_and_tokenizer(model_name):
    # Load the model and tokenizer based on the model name
    #if model_name == 'sbert':
        #model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
        #tokenizer = model.tokenizer
        
    #return model, tokenizer

model_name = 'sbert'
path = 'dataset/goodreads'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

# load the first stage processed data
filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl','rb'))

# construct book_dict: key book_id, value book_info_dict
book_dict = {}
for b in tqdm(filter_data_v5):
    assert b['book_id'] not in book_dict
    book_dict[b['book_id']] = b

#Creating dict for relation book to x(all possible nodes in domain)
book_to_book = {}
book_to_shelf = {}
book_to_author = {}
book_to_format = {}
book_to_publisher = {}
book_to_language_code = {}

# Populate the dictionaries
for book_id, book_info in book_dict.items():
    book_to_book[book_id] = book_info.get('similar_books', [])
    book_to_shelf[book_id] = [shelf['name'] for shelf in book_info.get('popular_shelves', [])]
    book_to_author[book_id] = [author['author_id'] for author in book_info.get('authors', [])]
    if book_info.get('format', '') != '':
        book_to_format[book_id] = [book_info.get('format', '')]
    if book_info.get('publisher', '') != '':
        book_to_publisher[book_id] = [book_info.get('publisher', '')]
    if book_info.get('language_code', '') != '':
        book_to_language_code[book_id] = [book_info.get('language_code', '')]

#Getting node-embedding
#model, tokenizer = load_model[model_name]()
model, tokenizer, device = load_model[model_name]()
#print(loaded_model)
text2embedding= load_text2embedding[model_name]
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Prepare data for embedding
descriptions = [book['title']+ " "+ book['description']  for book in book_dict.values()]

# Add these print statements to check the contents of descriptions
print("Number of descriptions:", len(descriptions))
print("Example description:", descriptions[3])

description_embeddings = text2embedding(model, tokenizer, device, descriptions)

# Add this print statement to check the length of description_embeddings
print("Number of description embeddings:", len(description_embeddings))

# Combine features for each book
book_features = {}
for book_id, desc_vec in zip(book_dict.keys(), description_embeddings):
    book_features[book_id] = desc_vec.numpy()
    
# Add this print statement to check the length of book_features
print("Number of books with features:", len(book_features))

# Stack the flattened feature vectors to create the feature matrix
book2feature = np.vstack(list(book_features.values()))

book2list = list(book_dict.keys())

# Check shapes
print("Number of books:", len(book_features))
print("Feature vector shape:", book2feature.shape)

def get_node_features(node_name, special_key=None):
    # Initialize a dictionary to hold the feature vectors for each node
    node_to_book = defaultdict(list)

    # Iterate over each book in the book_dict
    for book_id, book_data in book_dict.items():
        # Iterate over the nodes of the book
        if special_key:
            for node in book_data[node_name]:
                node_id = node.get(special_key)
                if node_id and book_id:
                    node_to_book[node_id].append(book_id)
        elif isinstance(book_data[node_name], list):
            for node in book_data[node_name]:
                if node and book_id:
                    node_to_book[node].append(book_id)
        elif book_data[node_name] and book_id:
            node_to_book[book_data[node_name]].append(book_id)

    # Step 2: Aggregate book features for each node
    node_features = {}
    for node, book_ids in node_to_book.items():
        # Filter out any book_ids that might not be in book_features
        valid_book_ids = [bid for bid in book_ids if bid in book_features]
        if valid_book_ids:
            combined_description = " ".join([book_dict[bid]['title'] + " " + book_dict[bid]['description'] for bid in valid_book_ids])
            node_features[node] = combined_description

    # Convert the node features to embeddings
    node_descriptions = list(node_features.values())
    node_description_vectors = text2embedding(model, tokenizer, device, node_descriptions)

    # Convert the embeddings into numpy arrays
    for node, embedding in zip(node_features.keys(), node_description_vectors):
        node_features[node] = embedding.numpy()

    # Stack the flattened feature vectors to create the feature matrix
    node2feature = np.vstack(list(node_features.values()))

    # Check shapes
    print("Number of nodes:", len(node_features))
    print("Feature vector shape:", node2feature.shape)
    node2list = list(node_to_book.keys())
    return node_to_book, node_features, node2feature, node2list

# Call get_node_features function for each node type
author_to_book, author_features, author2feature, author2list = get_node_features('authors', 'author_id')
book_to_book, book_features_combined, book2feature_combined, book2list_combined = get_node_features('similar_books')
shelf_to_book, shelf_features, shelf2feature, shelf2list = get_node_features('popular_shelves', 'name')
format_to_book, format_features, format2feature, format2list = get_node_features('format')
publisher_to_book, publisher_features, publisher2feature, publisher2list = get_node_features('publisher')
language_code_to_book, language_code_features, language_code_2feature, language2list = get_node_features('language_code')

def generate_graph_data():
    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    # Iterate over book data to generate nodes and edges
    for book_id, book_data in tqdm(book_dict.items()):
        nodes = {}
        edges = []
        
        # Authors
        for author in book_data.get('authors', []):
            author_id = author.get('author_id', '')
            if author_id not in nodes:
                nodes[author_id] = len(nodes)
                
        for author_id in book_data.get('authors', []):
            if author_id in nodes:
                edges.append({'src': nodes[author_id], 'edge_attr': 'author', 'dst': book_id})
        
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

# Call the function to generate nodes and edges
generate_graph_data()

def generate_split():
    # Define the size of each split (you can adjust these ratios as needed)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
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

if __name__ == '__main__':
    generate_split()
