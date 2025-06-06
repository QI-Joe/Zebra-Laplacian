import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from evaluation.evaluation import LogRegression, fast_eval_check
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data_TPPR
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
from utils.my_dataloader import to_cuda, Temporal_Splitting, Temporal_Dataloader, data_load
from itertools import chain
from modules.memory import EfficentMemory
from torch_geometric.data import Data as pygData
from torch_geometric.utils import get_laplacian

import warnings
from datetime import datetime
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

def str2bool(order: str)->bool:
  if order in ["True", "1"]:
    return True
  return False

parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='cora')
parser.add_argument('--bs', type=int, default=10000, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=7, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--snapshot', type=int, default=3, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.3, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')

parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--embedding_module', type=str, default="diffusion", help='Type of embedding module')

parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--save_best',action='store_true', help='store the largest model')
parser.add_argument('--tppr_strategy', type=str, help='[streaming|pruning]', default='streaming')
parser.add_argument('--topk', type=int, default=40, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')
parser.add_argument('--dynamic', type=str2bool, default=False)
parser.add_argument('--task', type=str, default="None", help='robustness task') # edge_disturb
parser.add_argument('--ratio', type=float, default=50, help='imbalance, few shot learning and edge distrubution ratio')
parser.add_argument('--cora_inductive', type=str2bool, default=False, help='whether to use inductive training')


parser.add_argument('--ignore_edge_feats', action='store_true')
parser.add_argument('--ignore_node_feats', action='store_true')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')

# python train.py --n_epoch 50 --n_degree 10 --n_layer 2 --bs 200 -d wikipedia --enable_random  --tppr_strategy streaming --gpu 0 --alpha_list 0.1 --beta_list 0.9

args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
USE_MEMORY = True
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
dynamic: bool = args.dynamic
ROBUST_TASK = args.task
CORA_INDUCTIVE: bool = args.cora_inductive
EPOCH_INTERVAL = 25
RATIO = args.ratio
SNAPSHOT = args.snapshot


round_list, graph_num, graph_feature, edge_num = get_data_TPPR(DATA, snapshot=SNAPSHOT, dynamic=dynamic, task=ROBUST_TASK, ratio=RATIO)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
training_strategy = "node"
NODE_DIM = round_list[0][0].node_feat.shape[1]
test_record = []
laplacian_memory = EfficentMemory(snapshot=SNAPSHOT, device=device, combination_method='sum')

all_run_times = time.time()
VIEW = len(round_list)

print(f"Given task is {ROBUST_TASK}")
for i in range(1):

  full_data, train_data, val_data, test_data, n_nodes, n_edges = round_list[i]
  num_classes = np.max(full_data.labels)+1

  args.n_nodes = graph_num +1
  args.n_edges = edge_num +1

  edge_feats = None
  node_feats = graph_feature
  node_feat_dims = full_data.node_feat.shape[1]

  if edge_feats is None or args.ignore_edge_feats: 
    print('>>> Ignore edge features')
    edge_feats = np.zeros((args.n_edges, 1))
    edge_feat_dims = 1

  train_ngh_finder = get_neighbor_finder(train_data)
  val_tppr_backup, test_tppr_backup = float(0), float(0)

  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_feats, edge_features=edge_feats, device=device,
            n_layers=NUM_LAYER,n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            node_dimension = NODE_DIM, time_dimension = TIME_DIM, memory_dimension=NODE_DIM,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            args=args)
  tgn.insert_laplacian_memory(laplacian_memory)
  projector = LogRegression(in_channels=128*3, num_classes=num_classes).to(device)

  criterion = torch.nn.BCELoss()
  criterion_node = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = torch.optim.Adam(chain(tgn.parameters(), projector.parameters()), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  t_total_epoch_train=0
  t_total_epoch_val=0
  t_total_epoch_test=0
  t_total_tppr=0
  stop_epoch=-1

  embedding_module = tgn.embedding_module

  print(f"the embedding module is {tgn.embedding_module_type}")

  # a variable name problem, train_src doesnt means np array contains source nodes
  # but both train sources and train destination nodes. 
  # which, train_src matches with batch_train
  train_src = np.concatenate([train_data.sources, train_data.destinations])
  timestamps_train = np.concatenate([train_data.timestamps, train_data.timestamps])

  embedding_module.streaming_topk_node(source_nodes=train_src, timestamps=timestamps_train, edge_idxs=train_data.edge_idxs)

  train_tppr_time, snapshot_list = [], []
  tppr_filled = False
  # TODO: Add laplacian embedding memory
  edge_index = torch.from_numpy(np.vstack((train_data.sources, train_data.destinations)))
  edge_index_lap, edge_weight_lap = get_laplacian(edge_index, normalization='sym', num_nodes=n_nodes)
  tgn.laplacian_memory.add_snapshot_memory(first_edge_idx_lap=edge_index_lap, first_edge_value_lap=edge_weight_lap, node_list=train_src)

  val_record = []
  for epoch in range(NUM_EPOCH):
    t_epoch_train_start = time.time()
    tgn.reset_timer()
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance/BATCH_SIZE)

    train_ap=[]
    train_acc=[]
    train_auc=[]
    train_loss=[]

    tgn.memory.__init_memory__()
    # tgn.set_neighbor_finder(train_ngh_finder)

    # model training
    tgn.train()
    projector.train()
    optimizer.zero_grad()

    for batch in range(num_batch):
      start = batch*BATCH_SIZE
      end = min((batch+1)*BATCH_SIZE, num_instance)
      batch_src = train_data.sources[start: end]
      batch_end = train_data.destinations[start:end]
      batch_time = train_data.timestamps[start:end]

      # a variable name problem, train_src doesnt means np array contains source nodes
      # but both train sources and train destination nodes. 
      # which, train_src matches with batch_train
      batch_train = np.concatenate([batch_src, batch_end])
      batch_train_time = np.concatenate([batch_time, batch_time])

      # node idx map, a very important step !!
      vector_map = np.vectorize(train_data.hash_table.get)
      node_emb = tgn.compute_temporal_node_embeddings(sources=batch_train, edge_times=batch_train_time, train=True)
      node_emb = projector.forward(node_emb)
    
      sample_node = np.array(list(set(batch_train)))
      sample_node = vector_map(sample_node)
      
      train_match_list, train_tranucated_label = train_data.robustness_match_tuple
      node_allow2see_mask = np.isin(sample_node, train_match_list)
      # node_allow2see_mask = np.ones_like(sample_node, dtype=bool)
      
      labels = train_data.labels[sample_node][node_allow2see_mask]
      labels_on_GPU = torch.tensor(labels).type(torch.LongTensor).to(device) # for label matching, it may need 2 mask...
      loss = criterion_node(node_emb[node_allow2see_mask], labels_on_GPU) # node_emb

      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

      with torch.no_grad():
        node_pred = node_emb[node_allow2see_mask].argmax(-1).cpu().numpy()
        train_ap.append(precision_score(labels.reshape(-1,1), node_pred.reshape(-1,1), average="macro", zero_division=1.0))
        train_acc.append(accuracy_score(labels.reshape(-1,1), node_pred.reshape(-1,1)))
        print(f"(TPPR) | snapshot {i} epoch {epoch} train ACC {train_acc[-1]:.5f}, train AP {train_ap[-1]:.5f}, Loss: {loss.item():.4f}")

    if (epoch+1) % EPOCH_INTERVAL != 0: continue
    tgn.eval()
    projector.eval()
    epoch_tppr_time = tgn.embedding_module.t_tppr
    train_tppr_time.append(epoch_tppr_time)

    epoch_train_time = time.time() - t_epoch_train_start
    t_total_epoch_train+=epoch_train_time
    train_ap=np.mean(train_ap)
    # train_auc=np.mean(train_auc)
    train_acc=np.mean(train_acc)
    train_loss=np.mean(train_loss)

    ########################  Model Validation on the Val Dataset #######################
    t_epoch_val_start=time.time()
    ### validation isolation, to cut off edge that gonna update during validation and later restore to train mode
    train_memory_backup = tgn.memory.backup_memory()
    train_tppr_backup = None
    if args.tppr_strategy=='streaming':
      train_tppr_backup = tgn.embedding_module.backup_tppr() # backup is a tuple involve (Norm_list, PPR_list)

    ### Frist, update the validation T-PPR ###
    val_source = np.concatenate([val_data.sources, val_data.destinations])
    val_timestamps = np.concatenate([val_data.timestamps, val_data.timestamps])
    # cache storage, imporve efficiency
    if val_tppr_backup == 0:
      embedding_module.streaming_topk_node(source_nodes=val_source, timestamps=val_timestamps, edge_idxs=val_data.edge_idxs)
    else:
      embedding_module.restore_tppr(val_tppr_backup)

    with torch.no_grad():
      val_emb = tgn.compute_temporal_node_embeddings(sources=val_source, edge_times=val_timestamps, train=False)
    
    val_check = fast_eval_check(val_source, val_data, val_emb, prj_model = projector)
    
    val_tppr_backup = embedding_module.backup_tppr()
    tgn.memory.restore_memory(train_memory_backup)
    embedding_module.restore_tppr(train_tppr_backup)

    epoch_val_time = time.time() - t_epoch_val_start
    t_total_epoch_val += epoch_val_time
    epoch_id = epoch+1

    ######################  Evaludate Model on the Test Dataset #######################
    t_test_start=time.time()

    ### transductive test
    val_memory_backup = tgn.memory.backup_memory()
    if args.tppr_strategy=='streaming':
      val_tppr_backup = tgn.embedding_module.backup_tppr()

    tgn.embedding_module.reset_tppr() # reset tppr to all 0
    test_source = np.concatenate([test_data.sources, test_data.destinations])
    test_timestamps = np.concatenate([test_data.timestamps, test_data.timestamps])
    
    if test_tppr_backup == 0:
      embedding_module.streaming_topk_node(source_nodes=test_source, timestamps=test_timestamps, edge_idxs=test_data.edge_idxs)
    else: 
      embedding_module.restore_tppr(test_tppr_backup)
    
    with torch.no_grad():
      test_emb = tgn.compute_temporal_node_embeddings(sources=test_source, edge_times=test_timestamps, train=False)
    
    test_check = fast_eval_check(test_source, test_data, test_emb, prj_model = projector)
    # store the test_tppr and save update cost
    test_tppr_backup = embedding_module.backup_tppr()
    """
    key in test_metrics:
    Transductive: val_acc, val_prec, val_recall, val_f1; test_acc, test_prec, test_recall, test_f1
    Inductive: nn_val_acc, nn_val_prec, nn_val_recall, nn_val_f1; nn_test_acc, nn_test_prec, nn_test_recall, nn_test_f1
    """

    tgn.memory.restore_memory(train_memory_backup)
    if args.tppr_strategy=='streaming':
      tgn.embedding_module.restore_tppr(train_tppr_backup)

    t_test=time.time()-t_test_start

    # train_tppr_time=np.array(train_tppr_time)[1:]

    NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH

    print('\n(TPPR) | snapshot {} epoch {} val acc: {:.4f}, val precision: {:.4f}, val recall: {:.4f}, val f1: {:.4f}'.format(i, epoch, \
        val_check["val_acc"], val_check["val_prec"], val_check["val_recall"], val_check["val_f1"]))
    print('(TPPR) | snapshot {} epoch {} test acc: {:.4f}, test precision: {:.4f}, test recall: {:.4f}, test f1: {:.4f}'.format(i, epoch, \
        test_check["val_acc"], test_check["val_prec"], test_check["val_recall"], test_check["val_f1"]))

    if "nn_val_acc" in val_check:
      print('(TPPR) | snapshot {} epoch {} nn_val acc: {:.4f}, nn_val precision: {:.4f}, nn_val recall: {:.4f}, nn_val f1: {:.4f}'.format(
        i, epoch, val_check["nn_val_acc"], val_check["nn_val_prec"], val_check["nn_val_recall"], val_check["nn_val_f1"]))
    if "nn_val_acc" in test_check:
      print('(TPPR) | snapshot {} epoch {} nn_test acc: {:.4f}, nn_test precision: {:.4f}, nn_test recall: {:.4f}, nn_test f1: {:.4f}'.format(
        i, epoch, test_check["nn_val_acc"], test_check["nn_val_acc"], test_check["nn_val_acc"], test_check["nn_val_acc"]))

    print(f'### num_epoch time {NUM_EPOCH}, epoch_train time {round(t_total_epoch_train/NUM_EPOCH, 4)}, epoch_val time {round(t_total_epoch_val/NUM_EPOCH, 4)}, epoch_test time {round(t_test, 4)}, train_tppr time {round(np.mean(train_tppr_time), 4)}')
    # print(f"### all epoch train time {round(t_total_epoch_train, 4)}, entire tppr finder time {round(np.sum(train_tppr_time), 4)}, entire run time without data loading: {round(time.time()-all_run_times, 4)}")

    snapshot_list.append((val_check, test_check))

    if not CORA_INDUCTIVE: continue

    val_record = np.array(val_record).T
    val_mean = np.mean(val_record, axis=1)

    test_node = set(test_data.sources) | set(test_data.destinations)
    val_node = set(val_data.sources) | set(test_data.destinations)
    train_node = set(train_data.sources) | set(train_data.destinations)

    assert len(train_node & val_node) == 0, f"train node and val node has interaction, which are {train_node & test_node}"
    assert len(train_node & val_node & test_node) == 0, f"test node, train node and val node has interaction"
    all_nodes = train_node | val_node | test_node
    if len(all_nodes) != 2708:
      print(f"Not All node is considered, number of nodes: {len(all_nodes)}")
      print(set([i for i in range(2708)]) - all_nodes)

    print(f"Number of test nodes: {len(test_node)}")
    print(f"Number of validation nodes: {len(val_node)}")
    print(f"Number of training nodes: {len(train_node)}")
  
  # Compute mean values for each metric in val_check and test_check across all snapshots
  mean_val_metrics = {}
  mean_test_metrics = {}

  # Initialize dictionaries with keys from the first snapshot
  for key in snapshot_list[0][0].keys():
    mean_val_metrics[key] = 0.0
  for key in snapshot_list[0][1].keys():
    mean_test_metrics[key] = 0.0

  # Accumulate values for each metric
  for val_check, test_check in snapshot_list:
    for key, value in val_check.items():
      mean_val_metrics[key] += value
    for key, value in test_check.items():
      mean_test_metrics[key] += value

  # Compute the mean by dividing by the number of snapshots
  num_snapshots = len(snapshot_list)
  for key in mean_val_metrics:
    mean_val_metrics[key] /= num_snapshots
  for key in mean_test_metrics:
    mean_test_metrics[key] /= num_snapshots

  test_record.append((mean_val_metrics, mean_test_metrics))


times = time.time()
# Convert the current time to day_hour_time_month format
formatted_time = datetime.fromtimestamp(times).strftime('%m_%d_%H_%M')
file_name = f"{DATA}_{SNAPSHOT}_{formatted_time}_{ROBUST_TASK}_{RATIO}"
with open(rf"log/{file_name}.txt", "w") as file:

  for idx, recs in enumerate(test_record):
    val, tests = recs
    print(f"At snapshot {idx}, we get:")
    file.write(f"At snapshot {idx}, we get:\n")
    
    for metric, value in val.items():
      print(f"avg {metric}: {value}")
      file.write(f"avg {metric}: {value}\n")
    
    print("\n")
    file.write("\n")
    
    for metric, value in tests.items():
      print(f"test {metric}: {value}")
      file.write(f"test {metric}: {value}\n")
    
    print("\n")
    file.write("\n")