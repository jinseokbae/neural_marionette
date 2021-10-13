import torch
import networkx as nx
import numpy as np
from copy import deepcopy

def process_affinity_glob(affinity, BIG_NUM=1e4):
    N, K, *_ = affinity.size()
    with torch.no_grad():
        influence = affinity.max(dim=0).values.squeeze(-1)  # (K, K)
        topk_indices = influence.topk(N, dim=-1).indices
        kypt_indices = torch.arange(0, K, dtype=torch.int64).to(affinity.device)[:, None]

        # binary adjacency
        A_bin = torch.zeros(K, K).to(affinity.device)
        A_bin[kypt_indices, topk_indices] = 1
        A_bin = torch.max(A_bin, A_bin.transpose(0, 1))  # (K, K)
        A_bin = A_bin.detach().cpu().numpy()

        # weighted dijkstra
        A_dijk = np.ones((K, K)) * BIG_NUM  # 1e2 is very big number for A_dijk

        # first computation of A_dijk
        pairs = np.stack(np.where(A_bin), axis=-1)  # (N, 2)
        G = nx.Graph()
        G.add_nodes_from(
            [(i) for i in range(K)]
        )
        G.add_edges_from(
            [(pair[0], pair[1]) for pair in pairs]
        )
        length = dict(nx.all_pairs_dijkstra_path_length(G))
        for k in length.keys():
            for kdot in length[k].keys():
                A_dijk[k, kdot] = length[k][kdot]

        # ensure connected_component is one
        if nx.number_connected_components(G) > 1:
            # ensure one connected component
            root = A_dijk.sum(axis=-1).argmin()
            priority = A_dijk.sum(axis=-1).copy().argsort()
            rank = np.zeros(K)
            for r, i in enumerate(priority):
                rank[i] = r
            candidates = np.where(A_dijk[root] == BIG_NUM)[0]  # (N, )
            min_idx = candidates[0]
            for candidate in candidates[1:]:
                if rank[min_idx] > rank[candidate]:
                    min_idx = candidate
            A_bin[root, min_idx] = 1
            A_bin[min_idx, root] = 1

            # # # weighted dijkstra
            A_dijk = np.ones((K, K)) * BIG_NUM  # 1e2 is very big number for A_dijk

            # first computation of A_dijk
            pairs = np.stack(np.where(A_bin), axis=-1)  # (N, 2)
            G = nx.Graph()
            G.add_nodes_from(
                [(i) for i in range(K)]
            )
            G.add_edges_from(
                [(pair[0], pair[1]) for pair in pairs]
            )
            length = dict(nx.all_pairs_dijkstra_path_length(G))
            for k in length.keys():
                for kdot in length[k].keys():
                    A_dijk[k, kdot] = length[k][kdot]

        # eliminate same score
        sum_dist = A_dijk.sum(axis=-1)
        A_bin_temp = deepcopy(A_bin)
        for k in range(K - 1):
            for kdot in range(k + 1, K):
                if sum_dist[k] == sum_dist[kdot]:
                    k_set = np.where(A_bin[k])[0]
                    kdot_set = np.where(A_bin[kdot])[0]
                    for n in k_set:
                        if n in kdot_set:
                            l = kdot if influence[n, k] > influence[n, kdot] else k
                            A_bin_temp[n, l] += 1e-5
                            A_bin_temp[l, n] += 1e-5

        A_dijk = np.ones((K, K)) * BIG_NUM  # 1e2 is very big number for A_dijk

        # first computation of A_dijk
        pairs = np.stack(np.where(A_bin), axis=-1)  # (N, 2)
        G = nx.Graph()
        G.add_nodes_from(
            [(i) for i in range(K)]
        )
        G.add_weighted_edges_from(
            [(pair[0], pair[1], A_bin_temp[pair[0], pair[1]]) for pair in pairs]
        )
        length = dict(nx.all_pairs_dijkstra_path_length(G))
        for k in length.keys():
            for kdot in length[k].keys():
                A_dijk[k, kdot] = length[k][kdot]

        A_dijk = torch.from_numpy(A_dijk).to(affinity.device)

        # finding parents
        priority = A_dijk.sum(dim=-1).topk(K, dim=-1, largest=False)
        root = priority.indices[0]
        priority = A_dijk[root].topk(K, largest=False)
        parents = []
        rank = A_dijk[root]
        for k in range(K):
            if k == root:  # root
                parent_idx = k
            else:
                neighbors = np.where(A_bin[k])[0]
                parent_idx = None
                parent_dist = -1e3
                for n in neighbors:
                    rank_dist = rank[n] - rank[k]
                    if rank_dist < 0 and rank_dist > parent_dist:
                        parent_dist = rank_dist
                        parent_idx = n
                    elif rank_dist < 0 and rank_dist == parent_dist:
                        if influence[k, n] > influence[k, parent_idx]:
                            parent_dist = rank_dist
                            parent_idx = n
                    elif rank_dist == 0:
                        n_neighbors = np.where(A_bin[n])[0]
                        co_parent_idx = None
                        co_parent_rank = 1e4
                        for nn in n_neighbors:
                            if nn in neighbors and rank[nn] < rank[n]:
                                if co_parent_rank > rank[nn]:
                                    co_parent_idx = nn
                                    co_parent_rank = rank[nn]
                        if co_parent_idx is not None:
                            if influence[co_parent_idx, n] > influence[co_parent_idx, k]:
                                parent_dist = rank_dist
                                parent_idx = n

                if parent_idx is None:
                    parent_idx = priority.indices[0]
                    A_bin[k, parent_idx] = 1
                    A_bin[parent_idx, k] = 1
            parents.append(parent_idx)
        parents = torch.LongTensor(parents).to(affinity.device)

        # make adjacency based on parent-child relationship
        A = torch.zeros_like(A_dijk)
        for k in range(K):
            if k == parents[k]:
                continue
            A[k, parents[k]] = 1
            A[parents[k], k] = 1

        # re-compute priority based on parent-child graph
        A_dijk = np.ones((K, K)) * BIG_NUM  # 1e2 is very big number for A_dijk

        pairs = np.stack(np.where(A.detach().cpu().numpy()), axis=-1)  # (N, 2)
        G = nx.Graph()
        G.add_nodes_from(
            [(i) for i in range(K)]
        )
        G.add_weighted_edges_from(
            [(pair[0], pair[1], A_bin_temp[pair[0], pair[1]]) for pair in pairs]
        )
        length = dict(nx.all_pairs_dijkstra_path_length(G))
        for k in length.keys():
            for kdot in length[k].keys():
                A_dijk[k, kdot] = length[k][kdot]

        A_dijk = torch.from_numpy(A_dijk).to(affinity.device)
        priority = A_dijk[root].topk(K, dim=-1, largest=False)

    return A, priority, parents