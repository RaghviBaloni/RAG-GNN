import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/dblp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

def generate_split():
    filter_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))
    paper_dict = {b['id']: b for b in tqdm(filter_data_v2)}

    paper2list = list(paper_dict.keys())
    labels = [[index, paper_dict[paper]['paper_label']] for index, paper in enumerate(paper2list)]

    label_labels = [label[1] for label in labels]
    dict_from_set = {element: index for index, element in enumerate(set(label_labels))}

    labels_v2 = [[index, dict_from_set[paper_dict[paper]['paper_label']]] for index, paper in enumerate(paper2list)]

    # Define fixed sizes for the splits
    num_train = 3500
    num_val = 500
    num_test = 2000

    labels_train = labels_v2[:num_train]
    labels_val = labels_v2[num_train:num_train + num_val]
    labels_test = labels_v2[num_train + num_val:num_train + num_val + num_test]
    labels_final = [labels_train, labels_val, labels_test]

    test_paper_ids = paper2list[num_train + num_val:num_train + num_val + num_test]

    os.makedirs(f'{path}/split', exist_ok=True)

    with open(f'{path}/split/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_train])))

    with open(f'{path}/split/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_val])))

    with open(f'{path}/split/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, [label[0] for label in labels_test])))

def step_one():
    filter_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))

    os.makedirs(path_nodes, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    def add_node(node_attr, node_dict):
        if node_attr not in node_dict:
            node_dict[node_attr] = len(node_dict)
        return node_dict[node_attr]

    def add_edge(src, edge_attr, dst, edge_list):
        edge_list.append({'src': src, 'edge_attr': edge_attr, 'dst': dst})

    for paper_data in tqdm(filter_data_v2):
        paper_id = paper_data['id']
        nodes = {}
        edges = []

        paper_node_id = add_node(paper_id, nodes)

        for author in paper_data.get('authors', []):
            author_name = author.get('name', '')
            author_org = author.get('org', '')
            author_id = author.get('id', '')

            if author_name:
                author_name_node = add_node(author_name, nodes)
                add_edge(paper_node_id, 'author_name', author_name_node, edges)

            if author_org:
                author_org_node = add_node(author_org, nodes)
                add_edge(paper_node_id, 'author_org', author_org_node, edges)

            if author_id:
                author_id_node = add_node(author_id, nodes)
                add_edge(paper_node_id, 'author_id', author_id_node, edges)

        year = paper_data.get('year', '')
        year_node = add_node(year, nodes)
        add_edge(paper_node_id, 'year', year_node, edges)

        n_citation = paper_data.get('n_citation', '')
        n_citation_node = add_node(n_citation, nodes)
        add_edge(paper_node_id, 'n_citation', n_citation_node, edges)

        doc_type = paper_data.get('doc_type', '')
        doc_type_node = add_node(doc_type, nodes)
        add_edge(paper_node_id, 'doc_type', doc_type_node, edges)

        publisher = paper_data.get('publisher', '')
        publisher_node = add_node(publisher, nodes)
        add_edge(paper_node_id, 'publisher', publisher_node, edges)

        doi = paper_data.get('doi', '')
        doi_node = add_node(doi, nodes)
        add_edge(paper_node_id, 'doi', doi_node, edges)

        for reference in paper_data.get('references', []):
            reference_node = add_node(reference, nodes)
            add_edge(paper_node_id, 'reference', reference_node, edges)

        venue = paper_data.get('venue', {}).get('raw', '')
        venue_node = add_node(venue, nodes)
        add_edge(paper_node_id, 'venue', venue_node, edges)

        abstract = paper_data.get('abstract', '')
        abstract_node = add_node(abstract, nodes)
        add_edge(paper_node_id, 'abstract', abstract_node, edges)

        nodes_df = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
        edges_df = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        nodes_df.to_csv(f'{path_nodes}/{paper_id}.csv', index=False)
        edges_df.to_csv(f'{path_edges}/{paper_id}.csv', index=False)

def step_two():
    filter_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))
    paper_dict = {b['id']: b for b in tqdm(filter_data_v2)}

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    print('Encoding paper abstracts...')
    abstracts = [paper['title'] + " " + paper['abstract'] for paper in paper_dict.values()]

    pbar = tqdm(total=len(abstracts))
    abstract_embeddings = []
    for abstract in abstracts:
        embedding = text2embedding(model, tokenizer, device, [abstract])
        abstract_embeddings.append(embedding)
        pbar.update(1)
    pbar.close()

    print("Number of abstract embeddings:", len(abstract_embeddings))

    paper_features = {paper_id: desc_vec.numpy() for paper_id, desc_vec in zip(paper_dict.keys(), abstract_embeddings)}
    print("Number of papers with features:", len(paper_features))

    paper2feature = np.vstack(list(paper_features.values()))
    paper2list = list(paper_dict.keys())

    print("Number of papers:", len(paper_features))
    print("Feature vector shape:", paper2feature.shape)

    a_embs = torch.tensor(paper2feature, dtype=torch.float)
    torch.save(a_embs, f'{path}/a_embs.pt')

    print('Encoding graphs...')
    os.makedirs(path_graphs, exist_ok=True)
    for paper_id in tqdm(paper_dict.keys()):
        nodes = pd.read_csv(f'{path_nodes}/{paper_id}.csv')
        edges = pd.read_csv(f'{path_edges}/{paper_id}.csv')

        nodes.node_attr.fillna("", inplace=True)

        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges['src'].tolist(), edges['dst'].tolist()])

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))
        torch.save(graph_data, f'{path_graphs}/{paper_id}.pt')

def generate_question_embeddings():
    filter_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))
    paper_dict = {b['id']: b for b in tqdm(filter_data_v2)}

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    questions = [f"Predict the research area (machine_learning, theoretical_computer_science, or computer_networking) of the paper '{paper['title']}' based on the details." for paper in paper_dict.values()]

    print('Encoding questions...')
    q_embs = text2embedding(model, tokenizer, device, questions)
    torch.save(q_embs, f'{path}/q_embs.pt')

if __name__ == '__main__':
    step_one()
    generate_question_embeddings()
    step_two()
    generate_split()

