from utils.my_dataloader import Temporal_Dataloader, NodeIdxMatching, Dynamic_Dataloader, data_load
import numpy as np
from numpy import ndarray
import copy
from typing import List, Tuple, Optional, Any
import torch

class Imbalance(object): 
    def __init__(self, ratio: float, train_ratio: float, val_ratio: Optional[float | None] = None, *args, **kwargs):
        super(Imbalance, self).__init__(*args, **kwargs)
        self.imbalance_ratio = ratio
        self.train_ratio = train_ratio
        self.seen_node_with_val: ndarray = None
        self.node_match_list: Optional[NodeIdxMatching | Any] = None
        self.val_ratio = val_ratio
        self.missing_label_exits: bool = False
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Imbalance Data Evaluation. In the node classification task, given a dataset G = (𝐺𝑖 , 𝑦𝑖 ), we simulate class imbalance by setting \n
        the proportions of training samples per class as {1, 1/2^𝛽 , 1/3^𝛽 , . . . , 1/|Y|^𝛽 }, where 𝛽 ∈ {0, 0.5, 1, 1.5, 2} controls the imbalance ratio. \n
        The num-ber of samples in the first class is fixed under all 𝛽 values.
        """
        nodes, label = data.my_n_id.node["index"].values, data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
        node_num, node_match_list = data.num_nodes, copy.deepcopy(data.my_n_id.node)
        seen_node, seen_node_label = nodes[:int(node_num*self.train_ratio)], label[:int(node_num*self.train_ratio)]
        
        # All for implmentation
        # available_node_list: list[np.ndarray] = self.pooling_check(seen_node, node_match_list)
        
        uniqclass, uniquenum = np.unique(seen_node_label, return_counts=True)
        if uniqclass[0] == -1 or uniqclass[0]==-1.0:
            self.missing_label_exits = True
            # -1 is the label for missing label node, so we remove it
            uniqclass, uniquenum = uniqclass[1:], uniquenum[1:]
            # Sort by frequency in descending order
            sorted_indices = np.argsort(-uniquenum)
            uniqclass, uniquenum = uniqclass[sorted_indices], uniquenum[sorted_indices]
        fixed_sample = uniquenum[0]
        sample_per_classes = [int(fixed_sample/((i+1)**self.imbalance_ratio)) for i in range(len(uniqclass))]
        
        selected_idx, outside_select = [], []
        for class_label, num_samples in zip(uniqclass, sample_per_classes):
            class_idx = np.where(seen_node_label == class_label)[0]
            if num_samples>0 and len(class_idx)>0:
                selected = np.random.choice(class_idx, min(num_samples, len(class_idx)), replace=False)
                not_selected = list(set(class_idx.tolist()) - set(selected.tolist()))
                selected_idx.extend(selected)
                outside_select.extend(not_selected)
        
        """
        Attention! There should be a assert to evaluate one thing:
        (np.array(selected_idx.extend(outside_select)) == seen_node).all() == True
        """
        train_mask, val_mask, nn_val_mask = np.zeros(node_num, dtype=bool), np.zeros(node_num, dtype=bool), np.zeros(node_num, dtype=bool)
        
        outside_select.extend(nodes[int(node_num*self.train_ratio):].tolist()) # here, extended
        nn_val_node_idx = np.array(outside_select)
        train_mask[np.array(selected_idx)], val_mask[int(node_num*self.train_ratio):], nn_val_mask[nn_val_node_idx] = True, True, True

        if self.val_ratio is not None:
            # bullshit, what Copilot wrote is all wrong
            val_mask = np.zeros(node_num, dtype=bool)
            val_mask[int(node_num*self.train_ratio):int(node_num*(self.train_ratio+self.val_ratio))] = True
            nn_val_node_idx_limit = nn_val_node_idx[nn_val_node_idx<=int(node_num*(self.train_ratio+self.val_ratio))]
            nn_val_mask = np.zeros(node_num, dtype=bool)
            nn_val_mask[nn_val_node_idx_limit] = True
        val_mask = val_mask & (label != -1)
        nn_val_mask = nn_val_mask & (label != -1) if self.missing_label_exits else nn_val_mask
        
        self.seen_node_with_val = np.hstack([node_match_list.values[selected_idx, 1], node_match_list.values[val_mask|nn_val_mask, 1]])
        self.seen_node = node_match_list.values[selected_idx, 1]
        # train_mask has no problem, but val_mask and nn_val_mask will contain node with label -1.
        data.train_val_mask_injection(train_mask, val_mask, nn_val_mask)
        
        return data
    
    def test_processing(self, t1_data: Temporal_Dataloader):
        t1_nodenum = t1_data.num_nodes
        t1_match_list: ndarray = t1_data.my_n_id.node.values # "index", "original node idx", "label"
        t1_label = t1_data.my_n_id.node["label"].values
        
        # seen_node = self.seen_node
        # t_match_list: ndarray = self.node_match_list.node.values # "index", "original node idx", "label"
        
        t1_unseen_node_mask = ~np.isin(t1_match_list[:, 1], self.seen_node)
        nn_test_mask = t1_unseen_node_mask & (t1_label != -1) if self.missing_label_exits else t1_unseen_node_mask
        
        t1_data.test_mask_injection(nn_test_mask)
        
        return t1_data
        
        
class Few_Shot_Learning(object):
    def __init__(self, fsl_num: int, val_ratio: Optional[float|None] = None, *args, **kwargs):
        super(Few_Shot_Learning, self).__init__(*args, **kwargs)
        self.fsl_num = fsl_num
        self.seen_node_with_val: ndarray = None
        self.val_ratio = val_ratio # val_ratio here equals val_ratio + train_ratio
        self.missing_label_exits: bool = False
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Few-shot Evaluation. Specifically, For graph classification \n
        tasks, given a training graph dataset G = {(𝐺𝑖 , 𝑦𝑖 )}, we set the \n
        number of training graphs per class as 𝛾 ∈ {10, 20, 30, 40, 50}. 
        """
        node_num, label = data.num_nodes, data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
        uniquclss, uniqunum = np.unique(label, return_counts=True)
        if uniquclss[0] == -1 or uniquclss[0]==-1.0:
            self.missing_label_exits = True
            # -1 is the label for missing label node, so we remove it
            uniquclss, uniqunum = uniquclss[1:], uniqunum[1:]
        training_data, node_match_list = [], data.my_n_id.node.values # "index", "original node idx", "label"
        
        for cls in uniquclss:
            class_indices = np.where(label == cls)[0]
            np.random.shuffle(class_indices)
            
            num_samples = min(self.fsl_num, len(class_indices))
            current_cls_selelcted_indices = class_indices[: num_samples]
            training_data.extend(current_cls_selelcted_indices)
            
        # self.seen_node_with_val = node_match_list[training_data, 1] # training data could direct retrieve on match list index to locate original node idx
        train_mask = np.zeros(node_num, dtype=bool)
        train_mask[training_data] = True
        val_mask, nn_val_mask = copy.deepcopy(~train_mask), copy.deepcopy(~train_mask) # Attention here, nn_val_mask will equal to val_mask

        if self.missing_label_exits:
            full_val_mask = ~train_mask
            full_val_mask = full_val_mask & (label != -1) # remove the node with label -1
            val_mask, nn_val_mask = copy.deepcopy(full_val_mask), copy.deepcopy(full_val_mask) # Attention here, nn_val_mask will equal to val_mask
        # val_mask[int(node_num*self.fsl_num):], nn_val_mask[int(node_num*self.fsl_num):] = False, False
        
        self.seen_node_with_val = np.hstack([node_match_list[training_data, 1], node_match_list[val_mask|nn_val_mask, 1]])
        self.seen_node = node_match_list[training_data, 1]
        
        data.train_val_mask_injection(train_mask, val_mask, nn_val_mask)
        return data
    
    def test_processing(self, t1_data: Temporal_Dataloader):
        t1_nodenum = t1_data.num_nodes
        t1_match_list: ndarray = t1_data.my_n_id.node.values # "index", "original node idx", "label"
        t1_label = t1_data.my_n_id.node["label"].values
        
        t1_unseen_node_mask = ~np.isin(t1_match_list[:, 1], self.seen_node)
        nn_test_mask = t1_unseen_node_mask & (t1_label != -1) if self.missing_label_exits else t1_unseen_node_mask
        
        t1_data.test_mask_injection(nn_test_mask)
        
        return t1_data