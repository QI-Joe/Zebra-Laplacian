import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)
import sys
import torch.nn as nn
from utils.data_processing import Data
from model.tgn_model import TGN
from typing import Union
from torch.optim import Adam
import random
from typing import Optional, Any
from utils.robustness_injection import Imbalance, Few_Shot_Learning

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size):

  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_acc = [], [], []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = data.n_interactions
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
 
    for batch_idx in range(num_test_batch):
      start_idx = batch_idx * TEST_BATCH_SIZE
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      sources_batch = data.sources[sample_inds]
      destinations_batch = data.destinations[sample_inds]
      timestamps_batch = data.timestamps[sample_inds]
      edge_idxs_batch = data.edge_idxs[sample_inds]


      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, n_neighbors, train = False)
      
      pos_prob=pos_prob.cpu().numpy() 
      neg_prob=neg_prob.cpu().numpy() 

      pred_score = np.concatenate([pos_prob, neg_prob])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])
      
      true_binary_label= np.zeros(size)
      pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1)

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_acc.append(accuracy_score(true_binary_label, pred_binary_label))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        torch.nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

def Simple_Regression(embedding: torch.Tensor, label: Union[torch.Tensor | np.ndarray], num_classes: int, \
                      num_epochs: int = 1500,  project_model=None, return_model: bool = False) -> tuple[float, float, float, float]:
    
    device = embedding.device
    if not isinstance(label, torch.Tensor):
        label = torch.LongTensor(label).to(device)
    linear_regression = LogRegression(embedding.size(1), num_classes).to(device) if project_model==None else project_model
    f = torch.nn.LogSoftmax(dim=-1)
    optimizer = Adam(linear_regression.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 0
    for epoch in range(num_epochs):
        linear_regression.train()
        optimizer.zero_grad()
        output = linear_regression(embedding)
        loss = loss_fn(f(output), label)

        loss.backward(retain_graph=False)
        optimizer.step()

        # if (epoch+1) % 1000 == 0:
        #     print(f'LogRegression | Epoch {epoch+1}: loss {loss.item():.4f}')

    with torch.no_grad():
        projection = linear_regression(embedding)
        y_true, y_hat = label.cpu().numpy(), projection.argmax(-1).cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=1.0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
        prec_micro, recall_micro, f1_micro = precision_score(y_true, y_hat, average='micro', zero_division=1.0), \
                                            recall_score(y_true, y_hat, average='micro'),\
                                            f1_score(y_true, y_hat, average='micro')
    if return_model:
        return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, linear_regression
    
    return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, None

def dict_merge(d1: dict, d2: dict, k):
    if not d1:
        return d2
    
    for key, val in d2.items():
        d1[key] = (d1[key]*(k-1) + d2[key]) / k
    return d1

def eval_tgb_task(tgn: TGN, projector_, evaluator, data, val_batches: dict):
    tgn.eval()

    record_list = list()
    for idx, bt_val in enumerate(val_batches):
        _, labels, bc_edge_mask = bt_val
        val_src = data.sources[bc_edge_mask]
        val_dst = data.destinations[bc_edge_mask]
        val_timestamp = data.timestamps[bc_edge_mask]

        val_ = np.concatenate([val_src, val_dst])
        val_time_eval = np.concatenate([val_timestamp, val_timestamp])

        with torch.no_grad():
            val_emb = tgn.compute_node_probabilities(sources=val_, edge_times=val_time_eval, train=False)
            linkreg_ = projector_.forward(val_emb)

        labels_data = labels.numpy()
        eval_method = {"y_pred": linkreg_, 
                        "y_true": labels_data, 
                        "eval_metric": ["ndcg", "rmse"]}
        result = evaluator.eval(eval_method)
        ndcg10_score = result.get("ndcg", None)
        rmse_score = result.get("rmse", None)
        record_list.append([ndcg10_score, rmse_score])
    metrics_res = np.array(record_list)

    return metrics_res.sum(axis=1)

task_model: Union[LogRegression| None] = None
def eval_node_classification(tgn: TGN, num_classes: int, val_src, val_edge_time, val_data, prj_model):
  task_model = prj_model
  val_sample = np.array(list(set(val_src)))
  vector_map = np.vectorize(val_data.hash_table.get)
  val_sample = vector_map(val_sample)
  tgn.eval()

  with torch.no_grad():
    val_emb = tgn.compute_node_probabilities(sources=val_src, edge_times=val_edge_time, train=False)
  if isinstance(val_data.labels, list):
    val_data.labels = np.array(val_data.labels)
  val_label = val_data.labels[val_sample]
  eval_res, task_model = Simple_Regression(embedding=val_emb, label=val_label, num_classes=num_classes, project_model=task_model, return_model=True)
  return eval_res

def full_evaluation_method(output, label):
    acc = accuracy_score(label, output)
    prec = precision_score(label, output, average='macro', zero_division=1.0)
    rec = recall_score(label, output, average='macro', zero_division=1.0)
    f1 = f1_score(label, output, average='macro', zero_division=1.0)
    return acc, prec, rec, f1

def eval_node_classification_deprec(datas: tuple[Data, Data, Data], embs: tuple[torch.Tensor], num_classes: int, prj, train_epoches: int = 1000):
    global task_model
    train, val, test = datas
    val_emb, test_emb = embs
    f = nn.LogSoftmax(dim=-1)
    
    train_match_list, val_match_list, test_match_list = train.robustness_match_tuple, val.robustness_match_tuple, test.robustness_match_tuple
    nn_val_match_list, nn_test_match_list = train.inductive_match_tuple, test.inductive_match_tuple
    
    with torch.no_grad():
        val_node, val_label= val_match_list
        # validation emb, node and label match
        val_unique_nodes = val.unique_nodes
        vector_map = np.vectorize(val.hash_table.get)
        freshed_val_node = vector_map(sorted(val_unique_nodes))
        val_emb_mask = np.isin(freshed_val_node, val_node)
        val_label_mask = np.isin(val_node, freshed_val_node)
        
        val_output = prj(val_emb[val_emb_mask])
        
        if nn_val_match_list != None: 
            nn_val_node, nn_val_label = nn_val_match_list
            nn_val_emb_mask = np.isin(freshed_val_node, nn_val_node)
            nn_val_label_mask = np.isin(nn_val_node, freshed_val_node)
            nn_val_output = prj(val_emb[nn_val_emb_mask])
        
        test_node, test_label = test_match_list
        test_unqiue_nodes = test.unique_nodes
        test_vector_map = np.vectorize(test.hash_table.get)
        test_freshed_node = test_vector_map(sorted(test_unqiue_nodes))
        test_emb_mask = np.isin(test_freshed_node, test_node)
        test_label_mask = np.isin(test_node, test_freshed_node)
        
        test_output = prj(test_emb[test_emb_mask])
        
        if nn_test_match_list != None: 
            nn_test_node, nn_test_label = nn_test_match_list
            nn_test_emb_mask = np.isin(test_freshed_node, nn_test_node)
            nn_test_label_mask = np.isin(nn_test_node, test_freshed_node)
            nn_test_output = prj(test_emb[nn_test_emb_mask])
        
    np_val, np_test = val_output.argmax(-1).cpu().numpy(), test_output.argmax(-1).cpu().numpy()
    val_acc, val_prec, val_recall, val_f1 = full_evaluation_method(np_val, val_label[val_label_mask])
    test_acc, test_prec, test_recall, test_f1 = full_evaluation_method(np_test, test_label[test_label_mask])
    val_dict = {"val_acc": val_acc, "val_prec": val_prec, "val_recall": val_recall, "val_f1": val_f1, "test_acc": test_acc, "test_prec": test_prec, "test_recall": test_recall, "test_f1": test_f1}
    
    inductive_dict = dict()
    if nn_val_match_list != None:
        np_nn_val = nn_val_output.argmax(-1).cpu().numpy()
        nn_val_acc, nn_val_prec, nn_val_recall, nn_val_f1 = full_evaluation_method(np_nn_val, nn_val_label[nn_val_label_mask])
        inductive_dict = {**inductive_dict, **{"nn_val_acc": nn_val_acc, "nn_val_prec": nn_val_prec, "nn_val_recall": nn_val_recall, "nn_val_f1": nn_val_f1}}
    if nn_test_match_list != None:
        np_nn_test = nn_test_output.cpu().numpy()
        nn_test_acc, nn_test_prec, nn_test_recall, nn_test_f1 = full_evaluation_method(np_nn_test, nn_test_label[nn_test_label_mask])
        inductive_dict = {**inductive_dict, **{"nn_test_acc": nn_test_acc, "nn_test_prec": nn_test_prec, "nn_test_recall": nn_test_recall, "nn_test_f1": nn_test_f1}}
            
    return {**val_dict, **inductive_dict}, prj
        
        
def fast_eval_check(src_combination, data: Data, emb, prj_model):
    prj_model.eval()
    with torch.no_grad():
        output = prj_model(emb)
        
    val_sample_node = np.array(list(set(src_combination)))
    vector_transform = np.vectorize(data.hash_table.get)
    refreshed_node = vector_transform(val_sample_node)
    
    y_true = data.labels[refreshed_node]
    y_hat = output.argmax(-1).cpu().numpy()
    
    val_match_node, val_match_label = data.robustness_match_tuple
    nn_val_match = data.inductive_match_tuple
    
    from_selected_node2match_edge_node_mask = np.isin(refreshed_node, val_match_node)
    val_allow2see, val_label_allow2see = y_hat[from_selected_node2match_edge_node_mask], y_true[from_selected_node2match_edge_node_mask]
    
    if nn_val_match != None:
        nn_val_match_node, nn_val_match_label = nn_val_match
        nn_from_selected_node2match_edge_node_mask = np.isin(refreshed_node, nn_val_match_node)
        nn_val_allow2see, nn_val_label_allow2see = y_hat[nn_from_selected_node2match_edge_node_mask], y_true[nn_from_selected_node2match_edge_node_mask]
    
    val_acc, val_prec, val_recall, val_f1 = full_evaluation_method(val_allow2see, val_label_allow2see)
    val_dict = {"val_acc": val_acc, "val_prec": val_prec, "val_recall": val_recall, "val_f1": val_f1}
    if nn_val_match != None:
        nn_val_acc, nn_val_prec, nn_val_recall, nn_val_f1 = full_evaluation_method(nn_val_allow2see, nn_val_label_allow2see)
        val_dict = {**val_dict, **{"nn_val_acc": nn_val_acc, "nn_val_prec": nn_val_prec, "nn_val_recall": nn_val_recall, "nn_val_f1": nn_val_f1}}
    return val_dict
        