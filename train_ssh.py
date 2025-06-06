import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction, eval_node_classification, LogRegression
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data_TPPR
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
from utils.my_dataloader import to_cuda, Temporal_Splitting, Temporal_Dataloader, data_load
from itertools import chain
from utils.uselessCode import node_index_anchoring

import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

def str2bool(order: str)->bool:
  if order in ["True", "1"]:
    return True
  return False

parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='dblp')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=20, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=20, help='Number of runs')
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
parser.add_argument('--topk', type=int, default=20, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')
parser.add_argument('--dynamic', type=str2bool, default=False)


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


round_list, graph_num, graph_feature, edge_num = get_data_TPPR(DATA, snapshot=args.n_runs, dynamic=dynamic, task=None)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
training_strategy = "node"
NODE_DIM = round_list[0][0].node_feat.shape[1]
test_record = []


all_run_times = time.time()
for i in range(len(round_list)):

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
  val_tppr_backup = float(0)

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

  train_src = np.concatenate([train_data.sources, train_data.destinations])
  timestamps_train = np.concatenate([train_data.timestamps, train_data.timestamps])

  embedding_module.streaming_topk_node(source_nodes=train_src, timestamps=timestamps_train, edge_idxs=train_data.edge_idxs)

  train_tppr_time=[]
  tppr_filled = False

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
    tgn = tgn.train()
    projector = projector.train()
    optimizer.zero_grad()

    for batch in range(num_batch):
      start = batch*BATCH_SIZE
      end = min((batch+1)*BATCH_SIZE, num_instance)
      batch_src = train_data.sources[start: end]
      batch_end = train_data.destinations[start:end]
      batch_time = train_data.timestamps[start:end]

      batch_train = np.concatenate([batch_src, batch_end])
      batch_train_time = np.concatenate([batch_time, batch_time])
    
      node_emb = tgn.compute_node_probabilities(sources=batch_train, edge_times=batch_train_time, train=True)
      node_emb = projector.forward(node_emb)
    
      sample_node = np.array(list(set(batch_train)))
      vector_map = np.vectorize(train_data.hash_table.get)
      sample_node = vector_map(sample_node)

      labels = train_data.labels[sample_node]
      labels_on_GPU = torch.tensor(labels).type(torch.LongTensor)
      labels_on_GPU = labels_on_GPU.to(device)
      loss = criterion_node(node_emb, labels_on_GPU)
      
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

      with torch.no_grad():
        node_pred = node_emb.argmax(-1).cpu().numpy()
        train_ap.append(precision_score(labels.reshape(-1,1), node_pred.reshape(-1,1), average="macro", zero_division=1.0))
        train_acc.append(accuracy_score(labels.reshape(-1,1), node_pred.reshape(-1,1)))
        print(f"(TPPR) | snapshot {i} epoch {epoch} train ACC {train_acc[-1]:.5f}, train AP {train_ap[-1]:.5f}, Loss: {loss.item():.4f}")

    if (epoch+1) % 50 == 0:
      epoch_tppr_time = tgn.embedding_module.t_tppr
      train_tppr_time.append(epoch_tppr_time)
      tgn.eval()
      projector.eval()

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
        train_tppr_backup = tgn.embedding_module.backup_tppr()

      ### Frist, update the validation T-PPR ###
      val_source = np.concatenate([val_data.sources, val_data.destinations])
      val_timestamps = np.concatenate([val_data.timestamps, val_data.timestamps])
      if val_tppr_backup == 0:
        embedding_module.streaming_topk_node(source_nodes=val_source, timestamps=val_timestamps, edge_idxs=val_data.edge_idxs)
      else:
        embedding_module.restore_tppr(val_tppr_backup)

      val_metrics = eval_node_classification(tgn=tgn, num_classes=num_classes, val_data = val_data, val_src=val_source,\
                                                  val_edge_time=val_timestamps, prj_model=projector)
      
      val_record.append([val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"]])
      val_tppr_backup = embedding_module.backup_tppr()

      tgn.memory.restore_memory(train_memory_backup)
      embedding_module.restore_tppr(train_tppr_backup)

      epoch_val_time = time.time() - t_epoch_val_start
      t_total_epoch_val += epoch_val_time
      epoch_id = epoch+1
      print('\n(TPPR) | snapshot {} epoch {} val acc: {:.4f}, val precision: {:.4f} val recall: {:.4f} val f1: {:.4f}'.format(i, epoch, \
                val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"]))

  ######################  Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()

  ### transductive test
  val_memory_backup = tgn.memory.backup_memory()
  if args.tppr_strategy=='streaming':
    val_tppr_backup = tgn.embedding_module.backup_tppr()


  tgn.embedding_module.reset_tppr() # reset tppr to all 0
  test_source = np.concatenate([test_data.sources, test_data.destinations])
  test_timestamps = np.concatenate([test_data.timestamps, test_data.timestamps])
  embedding_module.streaming_topk_node(source_nodes=test_source, timestamps=test_timestamps, edge_idxs=test_data.edge_idxs)

  test_metrics = eval_node_classification(tgn=tgn, num_classes=num_classes, val_data = test_data, val_src=test_source,\
                                                  val_edge_time=test_timestamps, prj_model = projector)

  tgn.memory.restore_memory(val_memory_backup)
  if args.tppr_strategy=='streaming':
    tgn.embedding_module.restore_tppr(val_tppr_backup)

  t_test=time.time()-t_test_start

  train_tppr_time=np.array(train_tppr_time)[1:]

  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH

  print(f'### num_epoch time {NUM_EPOCH}, epoch_train time {round(t_total_epoch_train/NUM_EPOCH, 4)}, epoch_val time {round(t_total_epoch_val/NUM_EPOCH, 4)}, epoch_test time {round(t_test, 4)}, train_tppr time {round(np.mean(train_tppr_time), 4)}')
  print(f"### all epoch train time {round(t_total_epoch_train, 4)}, entire tppr finder time {round(np.sum(train_tppr_time), 4)}, entire run time without data loading: {round(time.time()-all_run_times, 4)}")
  print('\nTest statistics -- snapshot {} acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}\n\n'.format(i, test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]))

  val_record = np.array(val_record).T
  val_mean = np.mean(val_record, axis=1)

  test_record.append((val_mean, test_metrics))

times = time.time()
with open(rf"log/{args.data}_{args.n_runs}_{round(times, 4)}.txt", "w") as file:

  for idx, recs in enumerate(test_record):
    val, tests = recs
    test_acc, test_prec, test_recall, test_f1 = tests["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]
    print(f"""At snapshot {idx}, we get \navg acc {val[0]}, avg precision {val[1]}, \navg recall {val[2]}, avg f1 is {val[3]}\n\ntest acc {test_acc}, test precision {test_prec},\ntest recall {test_recall}, test f1 {test_f1}""")
  
    output = f"""At snapshot {idx}, we get \navg acc {val[0]}, avg precision {val[1]}, \navg recall {val[2]}, avg f1 is {val[3]}\n\ntest acc {test_acc}, test precision {test_prec},\ntest recall {test_recall}, test f1 {test_f1}\n\n"""
    file.write(output)