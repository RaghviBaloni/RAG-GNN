import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
from src.dataset.utils.dblpretrieval import retrieval_via_pcst
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/dblp'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

default_answer = ''


class DBLPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        self.filetr_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))
        self.paper_dict = {b['id']: b for b in tqdm(self.filetr_data_v2)}
        self.text2embedding = load_text2embedding[model_name]
        self.model, self.tokenizer, self.device = load_model[model_name]()
        self.q_embs = torch.load(f'{path}/q_embs.pt')  # Load the question embeddings
        self.paper_ids = list(self.paper_dict.keys())  # Create a list of paper IDs


    def __len__(self):
        """Return the len of the dataset."""
        return len(self.paper_dict)

    def __getitem__(self, idx):
        paper_id = self.paper_ids[idx] #Use the index to get the paperID
        paper_data = self.paper_dict[paper_id]
        question = f"Question: Predict the research area (machine_learning, theoretical_computer_science or computer_networking), of the paper '{paper_data['title']}'.\n Prediction: \n Explanation: "
        graph = torch.load(f'{cached_graph}/{paper_id}.pt')
        desc = open(f'{cached_desc}/{paper_id}.txt', 'r').read()
        label =('|').join(paper_data['paper_label']).lower()
        
        return {
            'id': paper_id,
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
    filetr_data_v2 = pickle.load(open(f'{path}/dblp_v2.pkl', 'rb'))
    paper_dict = {b['id']: b for b in tqdm(filetr_data_v2)}
    paper_ids = list(paper_dict.keys())

    q_embs = torch.load(f'{path}/q_embs.pt')
    for paper_id, paper_data in tqdm(paper_dict.items()):
        if os.path.exists(f'{cached_graph}/{paper_id}.pt'):
            continue

        graph = torch.load(f'{path_graphs}/{paper_id}.pt')
        textual_nodes = pd.read_csv(f'{path_nodes}/{paper_id}.csv')
        textual_edges = pd.read_csv(f'{path_edges}/{paper_id}.csv')
        q_emb = q_embs[paper_ids.index(paper_id)]  # Use question embedding
        #topk = min(5, len(textual_nodes))
        #print(f"Number of nodes: {len(textual_nodes)}, topk: {topk}")
        subg, desc = retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=5, topk_e=5, cost_e=0.5)

        print(f"Length of subgraph: {len(subg)}")

        torch.save(subg, f'{cached_graph}/{paper_id}.pt')
        open(f'{cached_desc}/{paper_id}.txt', 'w').write(desc)


if __name__ == '__main__':
    
    preprocess()

    dataset = DBLPDataset()
    

    data = dataset[list(dataset.paper_dict.keys())[30]]
    for node_attr, paper_id in data.items():
        print(f'{node_attr}: {paper_id}')

#Trying to check for git
