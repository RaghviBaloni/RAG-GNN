import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=5, topk_e=5, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    # Node Prizes
    n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
    topk = min(topk, graph.num_nodes, n_prizes.size(0))
    _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
    n_prizes = torch.zeros_like(n_prizes)
    n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    
    # Edge Prizes
    e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
    topk_e = min(topk_e, e_prizes.unique().size(0))
    topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
    e_prizes[e_prizes < topk_e_values[-1]] = 0.0
    last_topk_e_value = topk_e
    for k in range(topk_e):
        indices = e_prizes == topk_e_values[k]
        value = min((topk_e - k) / sum(indices), last_topk_e_value)
        e_prizes[indices] = value
        last_topk_e_value = value * (1 - c)
    
    # Edge costs
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)
    
    edges_array = np.array(edges)
    edges_array = np.mod(edges_array, graph.num_nodes - 1)

    print("Length of costs array:", len(costs))
    print("Number of rows in edges array:", len(edges_array))
    print("Range of adjusted source node indices:", edges_array[:, 0].min(), edges_array[:, 0].max())
    print("Range of adjusted destination node indices:", edges_array[:, 1].min(), edges_array[:, 1].max())

    vertices, edges = pcst_fast(edges_array, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)
    
    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))
    
    #max_index = len(textual_nodes) - 1
    #selected_nodes = [i for i in selected_nodes if i <= max_index]

    # Get all edges connected to the selected nodes
    connected_edges = [i for i, (src, dst) in enumerate(graph.edge_index.T.tolist()) if src in selected_nodes or dst in selected_nodes]

    # Select edges that are either in the PCST result or connected to the selected nodes
    selected_edges = list(set(selected_edges).union(connected_edges))

    # If there are more than 5 edges, keep only the top 5 based on their prizes
    if len(selected_edges) > 5:
        edge_prizes = [e_prizes[i] for i in selected_edges]
        selected_edges = [x for _, x in sorted(zip(edge_prizes, selected_edges), key=lambda pair: pair[0], reverse=True)[:5]]
    
    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    mapping = {n: i for i, n in enumerate(selected_nodes)}
    
    if graph.x.nelement() > 0:
        x = graph.x[selected_nodes]
    else:
        x = torch.tensor([])

    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist() if i in mapping]
    dst = [mapping[i] for i in edge_index[1].tolist() if i in mapping]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc
