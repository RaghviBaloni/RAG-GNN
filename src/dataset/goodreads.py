import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from src.dataset.utils.goodreadsretrieval import retrieval_via_pcst
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/goodreads'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

default_answer = ''


class GoodreadsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
        self.book_dict = {b['book_id']: b for b in tqdm(self.filter_data_v5)}
        self.text2embedding = load_text2embedding[model_name]
        self.model, self.tokenizer, self.device = load_model[model_name]()
        #self.d_embs = torch.load(f'{path}/d_embs.pt')
        self.q_embs = torch.load(f'{path}/q_embs.pt')  # Load the question embeddings
        self.book_ids = list(self.book_dict.keys())  # Create a list of book IDs


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.book_dict)

    def __getitem__(self, idx):
        book_id = self.book_ids[idx] #Use the index to get the bookID
        book_data = self.book_dict[book_id]
        question = f"Question: Predict the genre (fiction, non-fiction or romance) of the book '{book_data['title']}'.\nAnswer: "
        graph = torch.load(f'{cached_graph}/{book_id}.pt')
        desc = open(f'{cached_desc}/{book_id}.txt', 'r').read()
        label =('|').join(book_data['genres']).lower()
        
        return {
            'id': book_id,
            'question': question,
            'graph': graph,
            'desc': desc,
            'label': label,
        }

    def get_idx_split(self):
        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

def preprocess():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    filter_data_v5 = pickle.load(open(f'{path}/books_filtered.pkl', 'rb'))
    book_dict = {b['book_id']: b for b in tqdm(filter_data_v5)}
    book_ids = list(book_dict.keys())

    q_embs = torch.load(f'{path}/q_embs.pt')
    for book_id, book_data in tqdm(book_dict.items()):
        if os.path.exists(f'{cached_graph}/{book_id}.pt'):
            continue

        graph = torch.load(f'{path_graphs}/{book_id}.pt')
        textual_nodes = pd.read_csv(f'{path_nodes}/{book_id}.csv')
        textual_edges = pd.read_csv(f'{path_edges}/{book_id}.csv')
        q_emb = q_embs[book_ids.index(book_id)]  # Use question embedding
        #topk = min(5, len(textual_nodes))
        #print(f"Number of nodes: {len(textual_nodes)}, topk: {topk}")
        subg, desc = retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=5, topk_e=5, cost_e=0.5)

        print(f"Length of subgraph: {len(subg)}")

        torch.save(subg, f'{cached_graph}/{book_id}.pt')
        open(f'{cached_desc}/{book_id}.txt', 'w').write(desc)


if __name__ == '__main__':
    
    preprocess()

    dataset = GoodreadsDataset()
    

    data = dataset[list(dataset.book_dict.keys())[30]]
    for node_attr, book_id in data.items():
        print(f'{node_attr}: {book_id}')

