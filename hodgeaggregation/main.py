from lib.libdata import *
from lib.libcnn import *
import argparse
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--seed', metavar='seed', dest='seed', default=1776, type=int, help='RNG seed')
parser.add_argument('--gpu_usage', metavar='gpu_usage', dest='gpu_usage', default=0.2, type=float, help='Proportion of GPU memory to use')
parser.add_argument('--output_prefix', metavar='output_prefix', dest='output_prefix', default='', type=str, help='Prefix for output files')
parser.add_argument('--num_classes', metavar='num_classes', dest='num_classes', default=5, type=int, help='Number of communities in PPM')
parser.add_argument('--size_communities', metavar='size_communities', dest='size_communities', default=20, type=int, help='Size of /each/ community in PPM')
parser.add_argument('--p', metavar='p', dest='p', default=0.8, type=float, help='Intracluster edge probability')
parser.add_argument('--q', metavar='q', dest='q', default=0.2, type=float, help='Intercluster edge probability')
parser.add_argument('--max_diffuse', metavar='max_diffuse', dest='max_diffuse', default=25, type=int, help='Upper bound on random node diffusion time')
parser.add_argument('--noise_energy', metavar='noise_energy', dest='noise_energy', default=0.0, type=float, help='Standard deviation of Gaussian noise to corrupt flow signals')
parser.add_argument('--corrupt_set', metavar='corrupt_set', dest='corrupt_set', default='both', choices=['both', 'train', 'valid'], type=str, help='Corrupt training set, validation set, or both')
parser.add_argument('--shift_operator', metavar='shift_operator', dest='shift_operator', default='hodge', choices=['hodge', 'linegraph', 'laplacian'], type=str, help='Aggregation operator, Laplacian for node-space aggregation CNN')
parser.add_argument('--smooth_filter', metavar='smooth_filter', dest='smooth_filter', default=True, type=bool, help='Decides if aggregation operator is low-pass')
parser.add_argument('--num_shifts', metavar='num_shifts', dest='num_shifts', default=128, type=int, help='Number of aggregation steps: pass 0 or negative to use number of edges')
parser.add_argument('--num_train', metavar='num_train', dest='num_train', default=10000, type=int, help='Training set size')
parser.add_argument('--num_valid', metavar='num_valid', dest='num_valid', default=2000, type=int, help='Validation set size')
parser.add_argument('--batch_size', metavar='batch_size', dest='batch_size', default=100, type=int, help='Training batch size')
parser.add_argument('--learning_rate', metavar='learning_rate', dest='learning_rate', default=0.0001, type=float, help='ADAM optimizer learning rate')
parser.add_argument('--epochs', metavar='epochs', dest='epochs', default=1000, type=int, help='Maximum number of training epochs')
parser.add_argument('--patience', metavar='patience', dest='patience', default=10, type=int, help='Number of epochs without improvement before early stop. If 0 or negative, dont do early stopping')
parser.add_argument('--dropout_rate', metavar='dropout_rate', dest='dropout_rate', default=0.5, type=float, help='Dropout rate for fully connected classifier')
parser.add_argument('--layer_depths', metavar='layer_depths', dest='layer_depths', default=[16,32], nargs='+', type=int, help='Sequence of layer depths for CNN: filter_counts 16 32')
parser.add_argument('--kernel_sizes', metavar='kernel_sizes', dest='kernel_sizes', default=[4,8], nargs='+', type=int, help='Sequence of kernel sizes for CNN: kernel_sizes 4 8')

args = parser.parse_args()

# collect parameters
seed = args.seed
gpu_usage = args.gpu_usage
output_prefix = args.output_prefix
num_classes = args.num_classes
size_communities = args.size_communities
p = args.p
q = args.q
max_diffuse = args.max_diffuse
noise_energy = args.noise_energy
corrupt_set = args.corrupt_set
shift_operator = args.shift_operator
smooth_filter = args.smooth_filter
num_shifts = args.num_shifts
num_train = args.num_train
num_valid = args.num_valid
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
patience = args.patience
layer_depths = args.layer_depths
kernel_sizes = args.kernel_sizes
dropout_rate = args.dropout_rate

if patience < 1:
    patience = epochs

# check passed parameters
assert gpu_usage > 0.0 and gpu_usage <= 1.0
assert num_classes > 0
assert size_communities > 0
assert p > 0.0 and q > 0.0 and p <= 1.0 and q <= 1.0
assert max_diffuse > 0
assert noise_energy >= 0
assert num_train > 0
assert num_valid > 0
assert batch_size > 0
assert learning_rate > 0
assert epochs > 0
assert len(layer_depths) == len(kernel_sizes)
assert (np.array(layer_depths) > 0).all()
assert (np.array(kernel_sizes) > 0).all()
assert dropout_rate >= 0.0 and dropout_rate <= 1.0

#################################################################

# random seeding
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

# limit GPU usage to 1/5
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
keras.backend.set_session(tf.Session(config=config))

# generate data

G, max_degree_nodes, sample_edges = SBM(num_classes, size_communities, p, q)
agg_edges = [sample_edges[0]]
agg_nodes = [max_degree_nodes[0]]

if num_shifts < 1:
    num_shifts = G.size()

A = scaledadjacencymatrix(G).todense()

labels = np.random.choice(num_classes, num_train + num_valid)
y = keras.utils.to_categorical(labels, num_classes)
y_tr, y_vld = (y[:num_train], y[num_train:])

print('Creating flow data')
source_nodes = [max_degree_nodes[c] for c in labels]
diffusion_times = np.random.choice(max_diffuse, num_train + num_valid)
flows = generateflows(G, A, max_diffuse, diffusion_times, source_nodes)
N = noise_energy * np.random.randn(*flows.shape)
if corrupt_set == 'train':
    N[:, num_train:] = 0
elif corrupt_set == 'valid':
    N[:, :num_train] = 0
flows += N

if shift_operator == 'hodge':
    S = scaledhodgelaplacian(G)
elif shift_operator == 'linegraph':
    S = scaledlinegraphlaplacian(G)
    flows = np.abs(flows)
elif shift_operator == 'laplacian':
    S = scaledlaplacianmatrix(G)
    pinvBT = np.linalg.pinv(incidencematrix(G).todense().T)
    flows = np.array(pinvBT @ flows)
    # calculate gradients from flows (this is silly but its the easiest way to integrate this experiment in)

if smooth_filter:
    # S = I - S
    S *= -1
    for i in range(S.shape[0]):
        S[i,i] += 1

print('Creating aggregated signals')
if shift_operator == 'laplacian':
    agg = aggregator(S, agg_nodes, num_shifts)
else:
    agg = aggregator(S, agg_edges, num_shifts)
X = np.transpose(agg @ flows, [2, 0, 1]) # (num_train + num_valid) x num_shifts x num observed edges
X_tr, X_vld = (X[:num_train], X[num_train:])

# model generation
history = AccuracyHistory()
earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
model = cnn_model(layer_depths, kernel_sizes,
                  dropout_rate=dropout_rate, learning_rate=learning_rate,
                  num_classes=num_classes)

# train!
model.fit(X_tr, y_tr,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_vld, y_vld),
          callbacks=[history, earlystopper])

training_score = model.evaluate(X_tr, y_tr, verbose=0)
validation_score = model.evaluate(X_vld, y_vld, verbose=0)

# save information and results

## parameters and final performance
params_file = open(f'results/{output_prefix}-params.txt','w')
for arg in vars(args):
    if arg != 'epochs':
        params_file.write(f'{arg} {getattr(args, arg)}\n')
    else:
        params_file.write(f'{arg} {len(history.train_loss)}/{getattr(args, arg)}\n')
        
params_file.write(f'Training Loss:       {training_score[0]}\n')
params_file.write(f'Training Accuracy:   {training_score[1]}\n')
params_file.write(f'Validation Loss:     {validation_score[0]}\n')
params_file.write(f'Validation Accuracy: {validation_score[1]}\n')
params_file.close()

## training, validation  loss, accuracy
tr_loss_file = open(f'./results/{output_prefix}-tr-loss.txt', 'w')
tr_acc_file = open(f'./results/{output_prefix}-tr-acc.txt', 'w')
val_loss_file = open(f'./results/{output_prefix}-val-loss.txt', 'w')
val_acc_file = open(f'./results/{output_prefix}-val-acc.txt', 'w')

for file_handler, training_data in zip([tr_loss_file, tr_acc_file,
                                        val_loss_file, val_acc_file],
                                       [history.train_loss, history.train_acc,
                                        history.val_loss, history.val_acc]):
    for index, item in enumerate(training_data):
        file_handler.write(f'{index}\t{item}\n')
    file_handler.close()
