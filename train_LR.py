import math
import time
import argparse
import torch
import numpy as np
from evaluation.evaluation import eval_tgb_task, eval_node_classification, LogRegression
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data_TPPR, TGB_load, batch_processor
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
from utils.my_dataloader import to_cuda, Temporal_Splitting, Temporal_Dataloader, data_load, position_encoding
from itertools import chain
from tgb.nodeproppred.evaluate import Evaluator
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from torch_geometric.loader import TemporalDataLoader
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

def str2bool(order: str)->bool:
  if order in ["True", "1"]:
    return True
  return False

parser = argparse.ArgumentParser('Self-supervised training with diffusion models')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='tgbn-trade')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
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
parser.add_argument('--topk', type=int, default=20, help='keep the topk neighbor nodes')
parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.1, 0.1], help='ensemble idea, list of alphas')
parser.add_argument('--beta_list', type=float, nargs='+', default=[0.05, 0.95], help='ensemble idea, list of betas')
parser.add_argument('--dynamic', type=str2bool, default=True)


parser.add_argument('--ignore_edge_feats', action='store_true')
parser.add_argument('--ignore_node_feats', action='store_true')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')
parser.add_argument('--task', type=str, default="node", help="define the loss function and task type")

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
TASK = args.task.lower()


device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
name = "tgbn-trade"
dataset = PyGNodePropPredDataset(name=name, root="datasets")
df, temporal_label, edge_side_feat = dataset.dataset.generate_processed_files()

train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

eval_metric = dataset.eval_metric
num_classes = dataset.num_classes
data = dataset.get_TemporalData()

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]


graph_num = max(data.src.max().item(), data.dst.max().item())
edge_num = data.src.shape[0]
data = data.to(device)

graph_feat = position_encoding(graph_num, emb_size=64).numpy()
train_data, val_data, test_data = TGB_load(train_data, val_data, test_data, graph_feat)

training_strategy = "node"
NODE_DIM = graph_feat.shape[1]
test_record = []
evaluator = Evaluator(name=DATA)

all_run_times = time.time()
for i in range(1):

  num_classes = dataset.num_classes

  args.n_nodes = graph_num +1
  args.n_edges = edge_num +1

  edge_feats = data.msg.cpu().numpy()
  node_feats = torch.from_numpy(graph_feat)
  node_feat_dims = graph_feat.shape[1]

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
  
  projector_link_reg = LogRegression(in_channels=128*3, num_classes=num_classes).to(device)

  criterion_link_reg = torch.nn.MSELoss()
  criterion_node = torch.nn.CrossEntropyLoss(reduction="mean")
  optimizer = torch.optim.Adam(chain(tgn.parameters(), projector_link_reg.parameters()), lr=LEARNING_RATE)
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

  train_tppr_time=[]
  tppr_filled = False

  val_record = []
  batches = batch_processor(temporal_label, train_data)

  
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
    optimizer.zero_grad()

    for idx, batch in enumerate(batches):
      # start = batch*BATCH_SIZE
      # end = min((batch+1)*BATCH_SIZE, num_instance)
      backprop_mask, labels, bc_edge_mask = batch

      batch_src = train_data.sources[bc_edge_mask]
      batch_end = train_data.destinations[bc_edge_mask]
      batch_time = train_data.timestamps[bc_edge_mask]

      # a variable name problem, train_src doesnt means np array contains source nodes
      # but both train sources and train destination nodes. 
      # which, train_src matches with batch_train
      batch_train = np.concatenate([batch_src, batch_end])
      batch_train_time = np.concatenate([batch_time, batch_time])

      # node idx map, a very important step !!
      # vector_map = np.vectorize(train_data.hash_table.get)
      # transferred_edges = vector_map(batch_train)
      node_emb = tgn.compute_node_probabilities(sources=batch_train, edge_times=batch_train_time, train=True)
      
      link_reg = projector_link_reg.forward(node_emb)
      labels_on_GPU = torch.tensor(labels).to(device)

      assert link_reg.shape[0] == backprop_mask.shape[0], \
      f"output shape is not matched, expected {backprop_mask.shape[0]} but get {link_reg.shape[0]}"
      
      loss = criterion_node(link_reg[backprop_mask], labels_on_GPU)

      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())

      with torch.no_grad():
        link_reg_pred = link_reg.cpu().numpy()
        eval_method = {"y_pred": link_reg_pred, 
                       "y_true": labels.cpu().numpy(), 
                       "eval_metric": ["ndcg", "rmse"]}
        result = evaluator.eval(eval_method)
        ndcg10_score = result.get("ndcg", 0.0)
        rmse_score = result.get("rmse", 0.0)
        print(f"(TPPR) | snapshot {i} epoch {epoch} NDCG 10 {ndcg10_score:.5f}, RMSE {rmse_score:.5f}, Loss: {loss.item():.4f}")

    if (epoch+1) % 50 == 0:
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
        train_tppr_backup = tgn.embedding_module.backup_tppr()

      ### Frist, update the validation T-PPR ###
      val_source = np.concatenate([val_data.sources, val_data.destinations])
      val_timestamps = np.concatenate([val_data.timestamps, val_data.timestamps])
      # cache storage, imporve efficiency
      if val_tppr_backup == 0:
        embedding_module.streaming_topk_node(source_nodes=val_source, timestamps=val_timestamps, edge_idxs=val_data.edge_idxs)
      else:
        embedding_module.restore_tppr(val_tppr_backup)

      val_batches = batch_processor(temporal_label, val_data)
      val_metrics = eval_tgb_task(tgn=tgn, projector_=projector_link_reg, evaluator=evaluator, \
                                  data = val_data, val_batches = val_batches)
      
      val_record.append(val_metrics)
      val_tppr_backup = embedding_module.backup_tppr()

      tgn.memory.restore_memory(train_memory_backup)
      embedding_module.restore_tppr(train_tppr_backup)

      epoch_val_time = time.time() - t_epoch_val_start
      t_total_epoch_val += epoch_val_time
      epoch_id = epoch+1
      print('\n(TPPR) | snapshot {} epoch {} val NDCG@10: {:.4f}, val RMSE: {:.4f}'.format(i, epoch, \
                val_metrics[0], val_metrics[1]))

  ######################  Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()

  ### transductive test
  # val_memory_backup = tgn.memory.backup_memory()
  # if args.tppr_strategy=='streaming':
  #   val_tppr_backup = tgn.embedding_module.backup_tppr()


  # tgn.embedding_module.reset_tppr() # reset tppr to all 0
  test_source = np.concatenate([test_data.sources, test_data.destinations])
  test_timestamps = np.concatenate([test_data.timestamps, test_data.timestamps])
  embedding_module.streaming_topk_node(source_nodes=test_source, timestamps=test_timestamps, edge_idxs=test_data.edge_idxs)

  test_batches = batch_processor(temporal_label, test_data)
  test_metrics = eval_tgb_task(tgn=tgn, projector_=projector_link_reg, evaluator=evaluator, \
                                  data = test_data, val_batches=test_batches)
      
  # tgn.memory.restore_memory(val_memory_backup)
  # if args.tppr_strategy=='streaming':
  #   tgn.embedding_module.restore_tppr(val_tppr_backup)

  t_test=time.time()-t_test_start

  train_tppr_time=np.array(train_tppr_time)[1:]

  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH

  print(f'### num_epoch time {NUM_EPOCH}, epoch_train time {round(t_total_epoch_train/NUM_EPOCH, 4)}, epoch_val time {round(t_total_epoch_val/NUM_EPOCH, 4)}, epoch_test time {round(t_test, 4)}, train_tppr time {round(np.mean(train_tppr_time), 4)}')
  print(f"### all epoch train time {round(t_total_epoch_train, 4)}, entire tppr finder time {round(np.sum(train_tppr_time), 4)}, entire run time without data loading: {round(time.time()-all_run_times, 4)}")
  print('\nTest statistics -- snapshot {} test NDCG@10: {:.4f}, test RMSE: {:.4f} \n\n'.format(i, test_metrics[0], test_metrics[1]))

  val_record = np.array(val_record).T
  val_mean = np.mean(val_record, axis=1)

  test_record.append((val_mean, test_metrics))

times = time.time()
with open(rf"log/{args.data}_{args.n_runs}_{round(times, 4)}.txt", "w") as file:

  for idx, recs in enumerate(test_record):
    val, tests = recs
    print(f"""At snapshot {idx}, we get \navg val NDCG@10 {val[0]}, avg val RMSE {val[1]} \n\ntest NDCG@10: {tests[0]}, test RMSE: {tests[1]}""")
    print(f"Test ")
    # test_acc, test_prec, test_recall, test_f1 = tests["accuracy"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"]
    output = f"""At snapshot {idx}, we get \navg val NDCG@10 {val[0]}, avg val RMSE {val[1]} \n\ntest NDCG@10: {tests[0]}, test RMSE: {tests[1]}\n\n"""
    file.write(output)