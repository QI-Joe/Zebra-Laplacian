from cgi import test
from pickle import TRUE
import random
import numpy as np
import numba as nb
from numba import types, typed
from numba.typed import listobject, dictobject
import sys
from sympy import Union, beta
from torch_geometric.data import Data
from scipy import sparse
import torch
from torch_geometric.utils import to_dense_adj
from utils.my_dataloader import data_load, Temporal_Dataloader, Temporal_Splitting, Dynamic_Dataloader
from typing import Union as union
import copy
import time
from sklearn.preprocessing import normalize



class Running_Permit:
    def __init__(self, event_or_snapshot: bool, ppr_updated: bool) -> None:
        self.event_or_snapshot = event_or_snapshot
        self.ppr_updated = ppr_updated

    def __call__(self, status: str, *args: np.ndarray, **kwds: np.ndarray) -> bool:
        if  self.event_or_snapshot=="event" and status=="extraction":
            return True
        elif self.event_or_snapshot=="event" and status=="updating":
            return False
        if self.event_or_snapshot=="snapshot" and self.ppr_updated:
            return True
        else:
            return False

    def snap_shot_update(self, updated: bool=False) -> None:
        if self.event_or_snapshot=="event":
            raise ValueError("Event wont be processed to confirm udpate.")
        self.ppr_updated = updated
        return 

class TPPR_Simple(object):
    def __init__(self, alpha_list: list[float], node_num: int, beta_list: list[float], topk: int):
        super(TPPR_Simple, self).__init__()

        self.nb_key_type=nb.typeof((1,1,0.1))
        self.nb_tppr_dict=nb.typed.Dict.empty(
                            key_type=self.nb_key_type,
                            value_type=types.float64,
                        )
        self.alpha_list = alpha_list
        self.k = topk
        self.n_tppr = len(self.alpha_list)
        self.beta_list = beta_list
        self.num_nodes = node_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_ppr = 2
        self.norm_list, self.PPR_list = self.reset_tppr()

    def reset_node_num(self, num: int):
        self.num_nodes = num

    def reset_tppr(self):
        norm_list: list = typed.List()
        PPR_list: list = typed.List()
        for _ in range(self.n_tppr):
            temp_PPR_list=typed.List()
            for _ in range(self.num_nodes):
                tppr_dict = nb.typed.Dict.empty(
                                key_type=self.nb_key_type,
                                value_type=types.float64,
                            )
                temp_PPR_list.append(tppr_dict)
            # are we going to change 
            norm_list.append(np.zeros(self.num_nodes, dtype=np.float64))
            PPR_list.append(temp_PPR_list)

        return norm_list, PPR_list

    def extract_streaming_tppr(self, tppr, current_timestamp, k, node_list, edge_idxs_list, delta_time_list, weight_list,position):
        if len(tppr)!=0:
            tmp_nodes=np.zeros(k,dtype=np.int32)
            tmp_edge_idxs=np.zeros(k,dtype=np.int32)
            tmp_timestamps=np.zeros(k,dtype=np.float32)
            tmp_weights=np.zeros(k,dtype=np.float32)

            for j,(key,weight) in enumerate(tppr.items()):
                edge_idx=key[0]
                node=key[1]
                timestamp=key[2]
                tmp_nodes[j]=node

                tmp_edge_idxs[j]=edge_idx
                tmp_timestamps[j]=timestamp
                tmp_weights[j]=weight

            tmp_timestamps=current_timestamp-tmp_timestamps
            node_list[position]=tmp_nodes
            edge_idxs_list[position]=tmp_edge_idxs
            delta_time_list[position]=tmp_timestamps
            weight_list[position]=tmp_weights


    def streaming_topk(self, source_nodes, timestamps, edge_idxs, updated: Running_Permit = None):
        premit_license = None
        if updated != None and  premit_license == None:
            premit_license = updated
        if premit_license == None and updated == None:
            raise ValueError("Please provide the updated status.")
        n_edges=len(source_nodes) // 3
        n_nodes=len(source_nodes)
        
        batch_node_list = []
        batch_edge_idxs_list = []
        batch_delta_time_list = []
        batch_weight_list = []

        if premit_license("extraction"):
            for _ in range(self.n_tppr):
                batch_node_list.append(np.zeros((n_nodes, self.k), dtype=np.int32)) 
                batch_edge_idxs_list.append(np.zeros((n_nodes, self.k), dtype=np.int32)) 
                batch_delta_time_list.append(np.zeros((n_nodes, self.k), dtype=np.float32)) 
                batch_weight_list.append(np.zeros((n_nodes, self.k), dtype=np.float32)) 

        ###########  enumerate tppr models ###########
        for index0,alpha in enumerate(self.alpha_list):
            beta = self.beta_list[index0]
            inter_norm_list= self.norm_list[index0]
            inter_PPR_list= self.PPR_list[index0]

        ###########  enumerate edge interactions ###########
            for i in range(n_edges):
                source=source_nodes[i]
                target=source_nodes[i+n_edges]
                fake=source_nodes[i+2*n_edges]
                timestamp=timestamps[i]
                edge_idx=edge_idxs[i]
                pairs=[(source,target),(target,source)] if source!=target else [(source,target)]

                ########### ! first extract the top-k neighbors and fill the list ###########
                if  premit_license("extraction"):
                    self.extract_streaming_tppr(inter_PPR_list[source], timestamp, self.k, batch_node_list[index0], batch_edge_idxs_list[index0], batch_delta_time_list[index0], batch_weight_list[index0],i)
                    self.extract_streaming_tppr(inter_PPR_list[target],timestamp, self.k, batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i+n_edges)
                    self.extract_streaming_tppr(inter_PPR_list[fake],timestamp, self.k, batch_node_list[index0],batch_edge_idxs_list[index0],batch_delta_time_list[index0],batch_weight_list[index0],i+2*n_edges)

                """
                iterate n_edges times only to update PPR list, however if list is already computed,
                extraction operation is enough; thus continune the loop
                """
                if premit_license("updating"):
                    continue
                
                if not (i+1)%1000: print(f"node updated to {i+1} edges")
                ############# ! then update the PPR values here #############
                for index,pair in enumerate(pairs):
                    s1=pair[0]
                    s2=pair[1]

                    ################# s1 side #################
                    if inter_norm_list[s1]==0:
                        t_s1_PPR = nb.typed.Dict.empty(
                                        key_type=self.nb_key_type,
                                        value_type=types.float64,
                                    )
                        scale_s2=1-alpha
                    else:
                        t_s1_PPR = inter_PPR_list[s1].copy()
                        last_norm= inter_norm_list[s1]
                        new_norm=last_norm*beta+beta
                        scale_s1=last_norm/new_norm*beta
                        scale_s2=beta/new_norm*(1-alpha)
                        for key, value in t_s1_PPR.items():
                            t_s1_PPR[key]=value*scale_s1     

                    ################# s2 side #################
                    if inter_norm_list[s2]==0:
                        t_s1_PPR[(edge_idx,s2,timestamp)]=scale_s2*alpha if alpha!=0 else scale_s2
                    else:
                        s2_PPR = inter_PPR_list[s2]
                        for key, value in s2_PPR.items():
                            if key in t_s1_PPR:
                                t_s1_PPR[key]+=value*scale_s2
                            else:
                                t_s1_PPR[key]=value*scale_s2
                
                        new_key = (edge_idx,s2,timestamp)
                        t_s1_PPR[new_key]=scale_s2*alpha if alpha!=0 else scale_s2

                    ####### exract the top-k items ########
                    updated_tppr=nb.typed.Dict.empty(
                        key_type=self.nb_key_type,
                        value_type=types.float64
                    )

                    tppr_size=len(t_s1_PPR)
                    if tppr_size<= self.k:
                        updated_tppr=t_s1_PPR
                    else:
                        keys = list(t_s1_PPR.keys())
                        values = np.array(list(t_s1_PPR.values()))
                        inds = np.argsort(values)[-self.k:]
                        for ind in inds:
                            key=keys[ind]
                            value=values[ind]
                            updated_tppr[key]=value

                    if index==0:
                        new_s1_PPR=updated_tppr
                    else:
                        new_s2_PPR=updated_tppr

                ####### update PPR_list and norm_list #######
                if source!=target:
                    inter_PPR_list[source]=new_s1_PPR
                    inter_PPR_list[target]=new_s2_PPR
                    inter_norm_list[source]=inter_norm_list[source]*beta+beta
                    inter_norm_list[target]=inter_norm_list[target]*beta+beta
                else:
                    inter_PPR_list[source]=new_s1_PPR
                    inter_norm_list[source]=inter_norm_list[source]*beta+beta

        return batch_node_list, batch_edge_idxs_list, batch_delta_time_list, batch_weight_list


# -------------------- start to build exact-PPR computation ------------------------

def normalize_adjacency_matrix(A: torch.Tensor, I: torch.Tensor):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde: torch.Tensor = A + I
    degrees = A_tilde.sum(axis=0)
    D = torch.diag(degrees, 0)
    D = D.to_sparse_coo().pow(-0.5)
    A_tilde_hat = D@(A_tilde)@(D)
    return A_tilde_hat

def nomralize_adjacency_matrix_L2(A: torch.Tensor):
    denominator = A.sum(dim=1)+1e-6
    return A / denominator

def create_propagator_matrix(graph: torch.Tensor, alpha, model: str = "exact") -> torch.Tensor:
    """
    Creating a propagation matrix.
    :param graph: expected to be edge_index with PyG data.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix Dense torch matrix /
    dict with indices and values for sparse multiplication.
    """
    dense_version = to_dense_adj(graph)
    A = torch.where(dense_version>0, torch.tensor(1), dense_version) # Adj function with value = 1
    if len(A.shape) > 2: A = A.squeeze()
    I = torch.eye(A.shape[0])
    # A_tilde_hat = normalize_adjacency_matrix(A, I)
    A_tilde_hat = nomralize_adjacency_matrix_L2(A)
    if model == "exact":
        propagator = I-(1-alpha)*A_tilde_hat
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator

def setup_propagator(edge_index, model: str="exact"):
    """
    Defining propagation matrix (Personalized Pagrerank or adjacency).
    """
    global alpha_list, device
    propagator = create_propagator_matrix(edge_index, alpha_list[0], model)

    if model == "exact":
        propagator = propagator.to(device)
    else:
        edge_indices = propagator["indices"].to(device)
        edge_weights = propagator["values"].to(device)
    
    return propagator

class Element(object):
    def __init__(self, node, idx) -> None:
        super(Element, self).__init__()
        self.node = node
        self.idx = idx

def paralle_matching(nodes, weights):
    weight_mask = np.argsort(weights, axis=1)
    tppr_node = np.take_along_axis(nodes, weight_mask, axis=1)[:, ::-1]
    return tppr_node

def matrix2dict(matrix: torch.Tensor):
    """
    :param matrix: PPR matrix in Tensor dense format.
    Will convert matrix to a numpy array that its index is source node and value is a list of neighbor node
    sorted by PPR weight
    """
    value_idx = matrix.nonzero().numpy()
    matrix = matrix.numpy()
    length = matrix.shape[0]
    idx, max_len, temp_list, weight_list = 0, 0, [], []
    while idx < length:
        temp_list.append(value_idx[value_idx[:, 0] == idx][:, 1].tolist())
        weight_list.append([matrix[*matrix_idx] for matrix_idx in value_idx[value_idx[:, 0]==idx]]) # * to release numpy array from 2d to index
        max_len = max(len(temp_list[-1]), max_len)
        idx += 1
    padded_weight_list = [sub_list + [0]*(max_len-len(sub_list)) for sub_list in weight_list]
    padded_list = [sub_list + [-1]*(max_len-len(sub_list)) for sub_list in temp_list]
    return paralle_matching(np.array(padded_list), np.array(padded_weight_list))

def scan2match(ppr, tppr_node, tppr_weight, test_mode: bool = False):
    if test_mode:
        ppr_node = ppr
    else:
        ppr_node = matrix2dict(ppr)
        tppr_node = paralle_matching(tppr_node, tppr_weight)
    return match_method(ppr_node, tppr_node)

def match_method(key_ppr, compare_ppr):
    """
    we assume that row length of key_ppr should at least larger or equal to compare_ppr
    """
    length = key_ppr.shape[0]
    recorder, weight_idx_recorder = [], []

    for src in range(length):
        par1, par2, border = key_ppr[src], compare_ppr[src], compare_ppr[src].shape[0]
        idx = np.empty((np.max(par1)+1, ), dtype=object)
        for list_idx, node in enumerate(par1):
            if node == -1: break
            idx[node] = Element(node, list_idx)
        status_list = np.zeros((len(par2), ), dtype=bool)
        longest, temp, i =0,0, 0
        while i < border:
            if par2[i] == -1: break
            if par2[i]<len(idx) and idx[par2[i]] != None:
                if i-1 >= 0 and par2[i-1]<len(idx): # idx[par2[i-1]] != None can be deleted
                    if idx[par2[i-1]] != None and status_list[i-1] and idx[par2[i]].idx - idx[par2[i-1]].idx == 1:
                        temp += 1
                    elif temp >= 1:
                        longest = max(longest, temp)
                        temp = 1
                        status_list = np.zeros((len(par2), ), dtype=bool)
                    status_list[i] = True
                else:
                    status_list[i] = True
                    temp += 1
            i += 1
        longest = max(longest, temp)
        if longest > 1:
            weight_idx_recorder.append([[src, sub_idx] for sub_idx in par2[status_list]])
        recorder.append(longest)
    return recorder, weight_idx_recorder

def tppr2matrix(tppr_node: np.ndarray, tppr_weight: np.ndarray)->union[torch.Tensor|np.ndarray]:
    indices_list = [[src, dst, tppr_weight[src, idx]] \
                    for src in range(tppr_node.shape[0]) \
                    for idx, dst in enumerate(tppr_node[src]) \
                        if dst>0 or tppr_weight[src, idx] > 0]
    indices_and_value = torch.Tensor(indices_list).T
    indices = indices_and_value[:2, :].to(torch.int64)
    simulate_ppr_value = indices_and_value[2, :].to(torch.float32)
    values = torch.ones((indices.shape[1], ), dtype=torch.float32)
    tppr_adj = torch.sparse_coo_tensor(indices, values, torch.Size([tppr_node.shape[0], tppr_node.shape[0]]))
    tppr_adj_with_weight_sparse = torch.sparse_coo_tensor(indices, simulate_ppr_value, torch.Size([tppr_node.shape[0], tppr_node.shape[0]]))

    return tppr_adj, tppr_adj_with_weight_sparse

def row_node_difference(ppr, tppr_node, tppr_weight, threshold: float = 0.8):
    """
    :param ppr: exact PPR adjacency matrix in Tensor
    :param tppr_node: top-k node list in numpy
    :param tppr_weight: top-k weight list in numpy
    """
    topk = tppr_node.shape[1]
    tppr_adj, tppr_sparse = tppr2matrix(tppr_node=tppr_node, tppr_weight=tppr_weight)

    matrix_or = ppr + tppr_adj
    row_index = torch.where(matrix_or>1, torch.tensor(1), matrix_or).sum(dim=1) / topk # test simlarity
    rows = torch.where(row_index >= threshold)[0]
    row_mask = row_index >= threshold

    selected_ppr, selected_tppr_node, selected_tppr_weight = ppr[row_mask], tppr_node[row_mask], tppr_weight[row_mask]

    general_weight_eval, full_eq = row_weight_difference(tppr_sparse, ppr)
    # weight_eval = row_weight_difference(value_idx, ppr, selected_tppr_weight)

    matched = None
    if rows.shape[0] != 0:
        matched, weight_idx = scan2match(selected_ppr, selected_tppr_node, selected_tppr_weight)
        tppr_dense = tppr_sparse.to_dense()
        selected_weight_diff = []
        for widx in weight_idx:
            inter_list = [abs(tppr_dense[*sub_widx].item() - ppr[*sub_widx].item()) / max(max(tppr_dense[*sub_widx].item(), ppr[*sub_widx].item()), 1e-6)\
                          for sub_widx in widx]
            selected_weight_diff.append(np.mean(inter_list))
        matched = torch.Tensor(matched) / topk
    return rows, matched, general_weight_eval, selected_weight_diff, full_eq

def ppr_matrix_comparsion(ppr1: torch.Tensor, ppr2: torch.Tensor, threshold: float = 0.8):
    """
    used for compare two PPR matrix, which first compute row differnce then manage to 
    convert to same weight difference
    :param ppr1: PPR matrix in Tensor format
    """
    ppr1, ppr2 = ppr1.to_dense(), ppr2.to_dense()
    addication_matrix = ppr1 + ppr2
    diff_idx = addication_matrix != ppr1
    denominator = torch.where(ppr1>0, torch.tensor(1), ppr1).sum(dim=1)
    diff_ratio = diff_idx.sum(dim=1) / denominator
    qualified_rows = torch.where(diff_ratio >= threshold)[0]

    weight_diff, full_eq = row_weight_difference(ppr1, ppr2, True)

    ppr1_mask, ppr2_mask = ppr1[qualified_rows], ppr2[qualified_rows]
    ppr1_dict, ppr2_dict = matrix2dict(ppr1_mask), matrix2dict(ppr2_mask)
    denominator_ = denominator[qualified_rows]

    if qualified_rows.shape[0] != 0:
        matched, weight_idx = match_method(ppr1_dict, ppr2_dict)
        selected_weight_diff = []
        for widx in weight_idx:
            inter_list = [abs(ppr2[*sub_widx].item() - ppr1[*sub_widx].item()) / max(max(ppr2[*sub_widx].item(), ppr2[*sub_widx].item()), 1e-6)\
                          for sub_widx in widx]
            selected_weight_diff.append(np.mean(inter_list))
        matched = torch.Tensor(matched) / denominator_
    return qualified_rows, matched, weight_diff, selected_weight_diff, full_eq

def test_scan2match(ppr, tppr_node, tppr_weight):
    tppr_node = paralle_matching(tppr_node, tppr_weight)
    test_node = tppr_node[:10]
    fake_ppr = copy.deepcopy(test_node)
    np.random.seed(2024)
    for idx in range(1, 10):
        premute, modify = idx / 10, 1 - idx / 10
        dice = np.random.randint(0, 10) / 10
        if dice < premute:
            low, high = np.random.randint(0, 10, 2)
            if low>high: low, high = high, low
            np.random.shuffle(fake_ppr[idx][low:high])
        else:
            num = random.randint(1, 6)
            indices = np.random.choice(10, num, replace=False)
            fake_ppr[idx][indices] += np.random.randint(-1,6,size=(num,))
        # fake_ppr[idx] = np.concatenate([fake_ppr[idx], np.array([-1]*5)], axis=0)
    paded = np.full((10, 5), -1)
    fake_ppr = np.concatenate([fake_ppr, paded], axis=1)
    test_res, weight_idx = scan2match(fake_ppr, test_node, tppr_weight[:10], test_mode=True)
    return test_res, weight_idx

def row_weight_difference(tppr: torch.sparse, ppr: torch.Tensor, is_dense: bool = False):
    if not is_dense: tppr = tppr.to_dense()
    mask = (tppr!=0) & (ppr!=0)
    tppr_union, ppr_union = torch.zeros_like(tppr), torch.zeros_like(ppr) # to maintain the 2d matrix shape
    tppr_union[mask], ppr_union[mask] = tppr[mask], ppr[mask]
    tppr_avoid_nan = tppr_union.clone().masked_fill(tppr_union==0, 2)
    full_eq = torch.where(tppr_avoid_nan == ppr_union, torch.tensor(1), torch.tensor(0)).nonzero()
    diff_weight = torch.abs(tppr_union - ppr_union) / tppr_avoid_nan
    nomalizer = torch.tensor([torch.count_nonzero(row) for row in tppr])
    avg_diff_weight = diff_weight.sum(dim=1) / nomalizer
    return avg_diff_weight, full_eq

def node_index_anchoring(input_edge: union[torch.Tensor | np.ndarray]) -> torch.Tensor:
    """
    Select each node idx that the last time it appeared in a given combination of [src, dst, dst] \n
    Change idx at 'target' position at idx[0][target] to modify the number
    """
    if isinstance(input_edge, np.ndarray):
        input_edge = torch.from_numpy(input_edge)
    nodes = torch.unique(input_edge)
    first_occ = []
    for node in nodes:
        idx = torch.where(input_edge[:None] == node)
        first_occ.append(idx[0][-1].item())
    node_mask = torch.full((input_edge.shape[0], ), False)
    node_mask[first_occ] = True
    return node_mask

def Approx_personalized_pagerank_torch(adj_matrix: torch.Tensor, query, alpha: float=0.1, max_iter=100)->torch.Tensor:
    adj_matrix = adj_matrix.squeeze_()
    n = adj_matrix.shape[0]
    Q = Q_0 = torch.zeros((1, n))
    Q[0, query] = Q_0[0, query] = 1

    l1_matrix = torch.norm(adj_matrix, p=1, dim=1).reshape(-1,1)
    l1_matrix = l1_matrix.masked_fill(l1_matrix==0, 1e-6)
    adj_norm = (adj_matrix / l1_matrix).cuda()

    Q=Q_0 = Q.cuda()

    for _ in range(max_iter):
        Q = (1-alpha)*torch.mm(Q, adj_norm) + alpha*Q_0
    
    return Q.T.squeeze().cpu()

def Approx_personalized_pagerank(adj_matrix: torch.Tensor, query, alpha: float=0.1, max_iter=100)->np.ndarray:
    adj_matrix = adj_matrix.numpy().squeeze()
    n = adj_matrix.shape[0]
    Q=Q_0 = np.zeros((1,n))
    Q[0, query] = Q_0[0, query] = 1

    if len(adj_matrix.shape)>2:
        adj_matrix=adj_matrix.squeeze()

    adj_norm = normalize(adj_matrix, norm="l1", axis=1)

    for _ in range(max_iter):
        Q = (1-alpha)*np.dot(Q, adj_norm) + alpha*Q_0
    return Q.T.squeeze()


def tppr_matrix_computing(full_data: union[Data|Temporal_Dataloader], tppr: TPPR_Simple):
    permit = Running_Permit(event_or_snapshot="snapshot", ppr_updated=False)
    all_train_source, all_train_dest = full_data.sources, full_data.destinations
    all_train_time, all_train_edgeIdx = full_data.timestamps, full_data.edge_idxs
    # all_train_source, all_train_dest = full_data.ori_edge_index[0], full_data.ori_edge_index[1]
    # all_train_time, all_train_edgeIdx = np.array(range(0, full_data.ori_edge_index.shape[1])), np.array(range(0, full_data.ori_edge_index.shape[1]))

    input_source = np.concatenate([all_train_source, all_train_dest, all_train_dest])

    _, _, _, _ = tppr.streaming_topk(source_nodes=input_source, timestamps=all_train_time, edge_idxs=all_train_edgeIdx, updated=permit)
    permit.snap_shot_update(updated=True)
    
    selected_node, _, _, selected_weight = tppr.streaming_topk(source_nodes=input_source, timestamps=all_train_time, edge_idxs=all_train_edgeIdx, updated=permit)
    
    node_mask = node_index_anchoring(input_source)
    anchor_node, anchor_weight = selected_node[1][node_mask], selected_weight[1][node_mask]

    if anchor_node.shape[0] != anchor_weight.shape[0] != node_mask.shape[0]:
        raise ValueError("Anchor node length not right.")

    return anchor_node, anchor_weight

def tppr_querying(dataset: str, tppr, input_dt: Data = None):
    entire_graph = get_graph(dataset=dataset, split_ratio=input_dt)
    time2 = time.time()
    return time2, tppr_matrix_computing(entire_graph, tppr)


def ppr_comparsion(dataset: str, tppr: TPPR_Simple, single_call: bool = False):
    """
    compare exact-PPR reuslt with that of T-PPR
    """
    general_data, _ = data_load(dataset)
    snapshot = 3
    view = snapshot-2
    dataloader_list = Temporal_Splitting(general_data, snapshot).temporal_splitting()
    neighborloader = Dynamic_Dataloader(dataloader_list, general_data)

    for t in range(view):
        temporal = neighborloader.get_temporal()
        edge_index = torch.Tensor(temporal.edge_index).to(torch.int64)

        # pass number in PyG Data format as inital input
        # no need to sperate train and valid, will use the entire graph
        tppr_data: list = get_data_node(dataset_name=dataset, snapshot=snapshot, test_mode=temporal)
        full_data, full_train_data, full_val_data, test_data, extra_data, n_nodes, n_edges, node_label = tppr_data[0]

        anchor_node, anchor_weight = tppr_matrix_computing(full_data)

        if single_call and view<2:
            return anchor_node, anchor_weight

        propagator = setup_propagator(edge_index)
        qualify_row, coverage, weight_eval_row, matched_weight_diff, eq_weight = \
            row_node_difference(propagator.cpu(), anchor_node, anchor_weight)
        
        # for node difference analysis
        print("\n\nT-PPR and Exact-PPR comparison for node differnce:")
        print(f"qualified nodes: {qualify_row.shape[0]} \ncoverage: {coverage[coverage>0.1].shape[0]} \
              \nmax matched sub-list length: {torch.max(coverage[coverage>0.1]).item():03f} \ntotal nodes: {propagator.shape[0]}")
        
        # for weight difference analysis
        avg_weight = torch.mean(weight_eval_row).item()
        topk5 = torch.topk(weight_eval_row[weight_eval_row>0.], 5, largest=False)[0]
        assert len(matched_weight_diff) == coverage[coverage>0.1].shape[0], "Compute node and weight is not equal"

        print("\n\nT-PPR and Exact-PPR comparison for weight differnce:")
        print(f"average matrix weight difference: {avg_weight:03f} \nfirst 5 fit weight difference: {[round(val.item(), 5) for val in topk5]} \
              \nnumber of full weight equaled node: {eq_weight.shape[0]}\n")


if __name__ == "__main__":
    start_time = time.time()
    alpha_list, beta_list = [0.1, 0.1], [0.05, 0.95]
    topk, node_num = 10, 10_000
    tppr = TPPR_Simple(alpha_list=alpha_list, node_num=node_num, beta_list=beta_list, topk=topk)
    ppr_comparsion("Cora", tppr)
    print(f"Time cost: {time.time()-start_time:03f}, \nin minutes: {(time.time()-start_time)/60:03f}")


# for key, val in dfs.items():
#     val[timestamp] = val[timestamp].apply(lambda x: x - min_time)
#     if key == "sx-mathoverflow": continue
#     counter += 1
#     nodes = np.unique(val["src"]).tolist()
#     indices = node_label[node_label["node"].isin(nodes)].index
#     for idx in indices:
#         val = node_label.loc[idx, "label"]
#         if val :
#             if val <= 3: node_label.loc[idx, "label"] += counter+1
#             else: node_label.loc[idx, "label"] = 7
#         else:
#             node_label.label[idx] = counter


# train_sampler = MultiLayerNeighborSampler(fanouts)
# test_sampler = MultiLayerFullNeighborSampler(2)
# edges_time: List[float] = graph.edata["time"].tolist()
# max_time: float = max(edges_time)
# min_time: float = min(edges_time)
# span: float = max(edges_time) - min(edges_time)
# temporal_subgraphs: List[dgl.DGLGraph] = []
# nids: List[th.Tensor] = []
# train_dataloader_list: List[Tuple[NodeDataLoader, dgl.dataloading.DataLoader]] = []

# T: List[float] = sampling_layer(snapshots, views, span, strategy)
# snapshot_graph_list: List[np.array] = []
# for start in T:
#     end: float = min(start + span / snapshots, max_time)
#     sample_time: th.Tensor = (graph.edata['time'] <= end) # returns an bool value
#     # snapshot_graph_list.append(df_graph.loc[np.where(sample_time)[0], ["src", "dst"]])
#     snapshot_graph_list.append(np.where(sample_time)[0])

#     temporal_subgraph: dgl.DGLGraph = dgl.edge_subgraph(graph, sample_time, relabel_nodes=False) # U.subgraph是个脑瘫函数

#     temporal_subgraph = dgl.to_simple(temporal_subgraph)
#     temporal_subgraph = dgl.to_bidirected(temporal_subgraph, copy_ndata=True)
#     temporal_subgraph = dgl.add_self_loop(temporal_subgraph)

#     # ------- use of dgl shoud end at here, rest of part should be convert to PYG format ------------------------------------------------
#     nids.append(th.unique(temporal_subgraph.edges()[0]))
#     temporal_subgraphs.append(temporal_subgraph)

# train_nid_per_gpu: List[int] = list(reduce(lambda x, y: x&y, [set(nids[sg_id].tolist()) for sg_id in range(views)]))
# train_nid_per_gpu = random.sample(train_nid_per_gpu, batch_size)
# random.shuffle(train_nid_per_gpu)
# train_nid_per_gpu = th.tensor(train_nid_per_gpu)       

# train_dataloader_list: List[Tuple[NodeDataLoader, dgl.dataloading.DataLoader]] = []
# for sg_id in range(views):
#     train_dataloader: NodeDataLoader = NodeDataLoader(temporal_subgraphs[sg_id],
#                                         train_nid_per_gpu,
#                                         train_sampler,
#                                         batch_size=train_nid_per_gpu.shape[0],
#                                         shuffle=False,
#                                         drop_last=False,
#                                         num_workers=num_workers,
#                                         )
#     test_dataloader: dgl.dataloading.DataLoader = dgl.dataloading.DataLoader(temporal_subgraphs[sg_id],
#                                     temporal_subgraphs[sg_id].nodes(),
#                                     test_sampler,
#                                     batch_size=dataloader_size, # lower the batch size to 512
#                                     shuffle=False,
#                                     drop_last=False,
#                                     num_workers=num_workers,
#                     )
#     if sg_id == 0:
#         train_dataloader_list.append(([train_dataloader], test_dataloader))
#     else:
#         train_dataloader_list.append(([train_dataloader]+train_dataloader_list[-1][0], test_dataloader))

# ----------------- should be done on data loader part -------------------------------------------------------

    # here we own the dataset, but how to split it into train and test?

# def CLDG_eval(embedding_model, train_embs, train_labels, val_embs, val_labels, test_embs, test_labels, n_classes, device_id, trainTnum):
#     for _ in range(2): # 5
#         logreg = embedding_model(train_embs.shape[1], n_classes)
#         logreg = logreg.to(device_id)
#         loss_fn = nn.CrossEntropyLoss()
#         opt = th.optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-4)

#         best_val_acc, eval_micro, eval_weight = 0, 0, 0
#         for epoch in range(500):
#             logreg.train()
#             opt.zero_grad()
#             logits = logreg(train_embs)
#             preds = th.argmax(logits, dim=1)
#             train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
#             loss = loss_fn(logits, train_labels)
#             loss.backward()
#             opt.step()

#             logreg.eval()
#             with th.no_grad():
#                 val_logits = logreg(val_embs)
#                 test_logits = logreg(test_embs)

#                 val_preds = th.argmax(val_logits, dim=1)
#                 test_preds = th.argmax(test_logits, dim=1)

#                 val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                
#                 ys = test_labels.cpu().numpy()
#                 indices = test_preds.cpu().numpy()
#                 test_micro = th.tensor(f1_score(ys, indices, average='micro'))
#                 test_weight = th.tensor(f1_score(ys, indices, average='weighted'))

#                 if val_acc >= best_val_acc:
#                     best_val_acc = val_acc
#                     if (test_micro + test_weight) >= (eval_micro + eval_weight):
#                         eval_micro = test_micro
#                         eval_weight = test_weight
#             alldata.append([train_acc, val_acc, test_micro])
#         print('Total Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_micro:{:4f}, test_weight:{:4f}'.format(epoch+1, train_acc, val_acc, test_micro, test_weight))
#         micros.append(eval_micro)
#         weights.append(eval_weight)

#     micros, weights = th.stack(micros), th.stack(weights)
#     print('Linear evaluation Accuracy:{:.4f} on training time T1 - T{:03d}, Weighted-F1={:.4f}'.format(micros.mean().item(), trainTnum, weights.mean().item()))
#     return embedding_model, alldata

# def eval_node(
#         embeddings: th.Tensor,
#         dataset: Tuple, # (T(trian, valid), T+1(test))
#         dataset_name: str,
#         evaluator: nn.Module,
#         split: str = "0.4:0.6",
#         verbose: bool = False,
#         model_name = "GCA",
# ):
#     evaluation_metrics = dict()
#     train_ratio, val_ratio = map(float, split.split(':'))
#     if model_name == "CLDG":
#         if dataset_name == "dblp":
#             labels, train_idx, val_idx, test_idx, n_classes = CLDG_testdataloader(dataset_name, testIdx)
#         elif dataset_name == "mathoverflow":
#             nodes = testIdx
#             labels = idxloader.get_label_by_node(nodes)
#             labels = torch.tensor(labels)
#             lenth = len(nodes)
#             label_train, label_val, label_test = list(range(int(0.1*lenth))), list(range(int(0.1*lenth),int(0.2*lenth))), list(range(int(0.2*lenth), lenth))
#             train_idx, val_idx, test_idx = nodes[:int(0.1*lenth)], nodes[int(0.1*lenth):int(0.2*lenth)], nodes[int(0.2*lenth):]
#         ...
#     elif model_name == "GCA":
#         ...
#     elif model_name == "MVGRL":
#         ...
#     else: 
#         raise NotImplementedError("Model name not supported....")

#     return evaluation_metrics 


# def main_CLDG():
#     parser = argparse.ArgumentParser(description='CLDG')
#     parser.add_argument('--dataset', type=str, help="Name of the dataset.", default='dblp')
#     parser.add_argument('--n_classes', type=int, required=False, default=64)
#     parser.add_argument('--n_layers', type=int, default=2)
#     parser.add_argument('--snapshots', type=int, default=10)
#     parser.add_argument('--views', type=int, default=7)
#     args_dict = {
#         'hidden_dim': 128,
#         'fanout': '20,20',
#         'strategy': 'sequential',
#         'readout': 'max',
#         'batch_size': 256,
#         'dataloader_size': 512,
#         'GPU': 0,
#         'num_workers_per_gpu': 4,
#         'epochs': 15
#     }

#     for key, value in args_dict.items():
#         parser.add_argument(f'--{key}', type=type(value), default=value)
#     args = parser.parse_args()
#     # parse arguments
#     DATASET = args.dataset
#     HID_DIM = args.hidden_dim
#     N_CLASSES = args.n_classes
#     N_LAYERS = args.n_layers
#     FANOUTS = [int(i) for i in args.fanout.split(',')]
#     SNAPSHOTS = args.snapshots
#     VIEWS = args.views
#     STRATEGY = args.strategy
#     READOUT = args.readout
#     BATCH_SIZE = args.batch_size
#     DATALOADER_SIZE = args.dataloader_size
#     GPU = args.GPU
#     WORKERS = args.num_workers_per_gpu
#     EPOCHS = args.epochs

#     # output arguments for logging
#     print('Dataset: {}'.format(DATASET))
#     print('Hidden dimensions: {}'.format(HID_DIM))
#     print('number of hidden layers: {}'.format(N_LAYERS))
#     print('Fanout list: {}'.format(FANOUTS))
#     print('Batch size: {}'.format(BATCH_SIZE))
#     print('GPU: {}'.format(GPU))
#     print('Number of workers per GPU: {}'.format(WORKERS))
#     print('Max number of epochs: {}'.format(EPOCHS))


#     data, time_recorder = train_CLDG(dataset = DATASET, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=N_CLASSES,
#     fanouts=FANOUTS, snapshots=SNAPSHOTS, views=VIEWS, strategy=STRATEGY, readout=READOUT, 
#     batch_size=BATCH_SIZE, dataloader_size=DATALOADER_SIZE, num_workers=WORKERS, epochs=EPOCHS, GPU=GPU)

#     for i in range(len(data)):
#         view, epoch, loss_val, metrics = data[i]
#         temporal_T, epoch_T = time_recorder[i]
#         print(f'View {view}, \n \
#             min_loss {min(loss_val)}, \n \
#             Temporal Time {temporal_T:05f}, \n \
#             Avg Epoch Time {sum(epoch_T)/len(epoch_T):05f}, \n \
#             Train Acc {metrics["train_acc"]:05f}, \n \
#             Test Acc {metrics["test_acc"]:05f}, \n \
#             Val Acc {metrics["val_acc"]:05f} \n \
#             Avg accuracy {metrics["accuracy"]:05f}, \n \
#             Avg precision {metrics["precision"]:05f}, \n \
#             Avg recall {metrics["recall"]:05f}, \n \
#             Avg f1 {metrics["f1"]:05f}')