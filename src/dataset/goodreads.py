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

default_question = "Predict the genre (fiction, non-fiction or romance) of the book '{}'."
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
        self.d_embs = torch.load(f'{path}/d_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.book_dict)

    def __getitem__(self, book_id):
        book_data = self.book_dict[book_id]
        question = default_question.format(book_data['title'])
        graph = torch.load(f'{cached_graph}/{book_id}.pt')
        desc = open(f'{cached_desc}/{book_id}.txt', 'r').read()

        return {
            'id': book_id,
            'question': question,
            'graph': graph,
            'desc': desc,
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

    def preprocess(self):
        os.makedirs(cached_desc, exist_ok=True)
        os.makedirs(cached_graph, exist_ok=True)

        for book_id, book_data in tqdm(self.book_dict.items()):
            if os.path.exists(f'{cached_graph}/{book_id}.pt'):
                continue
            
            graph = torch.load(f'{path_graphs}/{book_id}.pt')
            nodes = pd.read_csv(f'{path_nodes}/{book_id}.csv')
            edges = pd.read_csv(f'{path_edges}/{book_id}.csv')
            d_emb = self.text2embedding(self.model, self.tokenizer, self.device, [book_data['title']])[0]  # Embedding for the book title         
               
            topk = min(3, len(nodes))
            print(f"Number of nodes: {len(nodes)}, topk: {topk}")
            subg, desc = retrieval_via_pcst(graph, d_emb, nodes, edges, topk=topk, topk_e=5, cost_e=0.5)
            print(f"Length of subgraph: {len(subg)}")

            torch.save(subg, f'{cached_graph}/{book_id}.pt')
            open(f'{cached_desc}/{book_id}.txt', 'w').write(desc)


if __name__ == '__main__':
    dataset = GoodreadsDataset()
    dataset.preprocess()

    data = dataset[list(dataset.book_dict.keys())[2400]]
    for node_attr, book_id in data.items():
        print(f'{node_attr}: {book_id}')
