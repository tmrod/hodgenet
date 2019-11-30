#!/usr/bin/python3

import numpy as np
from scipy.linalg import null_space
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import random
import os, sys

# Delete all flags and reset graph ----------
# Also make tensorflow shut up
# TODO: make fun of Santiago's IDE next week
# TODO: get made fun of for writing bad code

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

tf.reset_default_graph()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#np.random.seed(1776)
#tf.set_random_seed(1776)
#--------------------------------------------

# user-supplied flags ---------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('mask', 0.1, 'Mask percentage')
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('iters_per_epoch', 10, 'Number of steps per epoch')
flags.DEFINE_integer('layers', 6, 'Number of RNN layers')
flags.DEFINE_integer('training_size', 100, 'Number of generated examples')
flags.DEFINE_integer('batch_size', 8, 'Training batch size')
#flags.DEFINE_integer('early_stopping', 10, 'Early stopping count')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate')
flags.DEFINE_string('shift', 'hodge', 'Shift operator')
# hodge, linegraph, random, identity, unsigned
#------------------------------------------------------------------

# data loading ------------------------
f_file = 'flow.npy'
l_file = 'laplacian.npy'

f_np = np.load(f_file)
f_np /= np.max(np.abs(f_np))
f_np = f_np.astype(np.float32)
E = len(f_np)
#f_np = np.array([f_np]).T
#f_np = f_np.reshape(-1,1)

L1_np = np.load(l_file)
L1_pinv_np = np.linalg.pinv(L1_np)

E = L1_np.shape[0]
D = np.sum(np.abs(L1_np), axis=1) - 2
Dinv = np.diag(1/D)
D = np.diag(D)
sqrtDinv = np.sqrt(Dinv)

if FLAGS.shift=='linegraph':
    shift_np = -np.abs(L1_np)
    np.fill_diagonal(shift_np, np.diag(D))
    shift_np = shift_np*2/np.abs(np.max(np.linalg.eigvals(shift_np)))
else:
    shift_np = L1_np*2/np.abs(np.max(np.linalg.eigvals(L1_np)))

shift_np = shift_np.astype(np.float32)
#--------------------------------------

# nonlinear function ------------------------------
def DReLu(X, bias):
    #return tf.nn.relu(X-bias)
    #return X
    return tf.nn.relu(X+bias) - tf.nn.relu(-X+bias)
    #return tf.nn.relu(X)
#--------------------------------------------------

# trainable weights -------------------------------------
weight_init = tf.truncated_normal_initializer(stddev=0.05)
dim = 32
U = tf.get_variable('U',
                    shape=(E,dim),
                    initializer=weight_init)
V = tf.get_variable('V',
                    shape=(dim,dim),
                    initializer=weight_init)
W = tf.get_variable('W',
                    shape=(dim,1),
                    initializer=weight_init)
bh = tf.get_variable('bh',
                     dtype=tf.float32,
                     initializer=0.05)
bo = tf.get_variable('bo',
                     dtype=tf.float32,
                     initializer=0.)
#--------------------------------------------------------

# model definition --------------------------------------
input_tf = tf.placeholder(tf.float32)
input_layer = input_tf
hidden_layer = DReLu(tf.matmul(input_layer, U), bh)
for i in range(FLAGS.layers-1):
    input_layer = tf.matmul(shift_np, input_layer)
    hidden_layer = DReLu(tf.matmul(input_layer, U) \
                         + tf.matmul(hidden_layer, V), bh)
    output_intermediate = DReLu(tf.matmul(hidden_layer, W), bo)
    # ugly hack
    if(i==0):
        o0 = output_intermediate
    elif(i==1):
        o1 = output_intermediate
    elif(i==2):
        o2 = output_intermediate
    elif(i==3):
        o3 = output_intermediate
    hidden_layer = tf.layers.dropout(hidden_layer, rate=FLAGS.dropout)
output_layer = DReLu(tf.matmul(hidden_layer, W), bo)
#--------------------------------------------------------

# loss functions ------------------------------------------------------
ground_tf = tf.placeholder(tf.float32)
output_tf = tf.placeholder(tf.float32)
mask_tf = tf.placeholder(tf.float32)
#ignore_tf = tf.placeholder(tf.float32)
loss = tf.losses.mean_squared_error(output_layer*(1-mask_tf),
                                    ground_tf*(1-mask_tf)) \
                                    /tf.reduce_mean((1-mask_tf))
#loss_test = tf.losses.mean_squared_error(output_layer*(1-ignore_tf),
#                                         ground_tf*(1-ignore_tf)) \
#                                         /tf.reduce_mean(1-ignore_tf)
#train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate). \
#    minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate). \
    minimize(loss)
#----------------------------------------------------------------------

# data processing functions ------------------------------
# returns corrupted data, as well as mask
def corrupt(x, FLAGS):
    mask = np.random.rand(*x.shape)
    mask = (mask > FLAGS.mask).astype(np.float32)
    return mask*x, mask
# create a similar flow
# add cyclic noise
# add smooth gradient noise
# f + (I - invL @ L + invL) @ N
def new_flow(f, L, invL, corruption=0.3):
    N = corruption*np.random.randn(*f.shape)
    return f + (np.eye(f.shape[0]) - invL @ L + invL) @ N
#---------------------------------------------------------

# create training set ahead of time --------------------------
training_set = [new_flow(f_np, L1_np, L1_pinv_np)
                for _
                in range(FLAGS.training_size)]
#-------------------------------------------------------------

# tensorflow session -----------------------------------------
#true_f_np = collapse(f_np) # uncorrupted data
true_f_np = f_np # uncorrupted data
observed_f_np, observed_mask_np = corrupt(f_np, FLAGS) # corrupted data

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

costs = []
cost_epochs = []
tests = []
test_epochs = []

for e in range(FLAGS.epochs):
    ground_ = random.choice(training_set)
    input_ = observed_mask_np*ground_
    sess.run(train_op, feed_dict={input_tf: np.diag(input_),
                                    ground_tf: ground_.reshape(-1,1),
                                    mask_tf: observed_mask_np.reshape(-1,1)})
    cost = sess.run(loss, feed_dict={input_tf: np.diag(input_),
                                     ground_tf: ground_.reshape(-1,1),
                                     mask_tf: observed_mask_np.reshape(-1,1)})
    costs.append(cost)
    cost_epochs.append(e)
    print('Epoch\t{:d}\tCost\t{:f}\r'.format(e, cost), end='')
#-------------------------------------------------------------

print('\n')

# Plotting ---------------------------------------------------
#plt.plot(cost_epochs, costs, label='Training')
#plt.plot(test_epochs, tests, label='Testing')
#plt.legend()
#plt.title('{:s} shift operator'.format(FLAGS.shift))
#plt.savefig('results/{:s}_training.png'.format(FLAGS.shift))
#plt.gcf().clear()

test_flow = new_flow(true_f_np, L1_np, L1_pinv_np, 0.0)
#test_flow = new_flow(true_f_np, L1_np, L1_pinv_np)

mmm = observed_mask_np
mmm = 1-mmm
mmm = mmm.astype(np.bool)
print(mmm.shape)

final = sess.run(output_layer, feed_dict={input_tf: np.diag(observed_mask_np*test_flow)})
final = np.array(final).T[0]
final = final[mmm]
print(final.shape)
#final = final.T[0]

initial = np.copy(test_flow)
initial = initial[mmm]
print(initial.shape)
#initial = initial.T[0]

plt.scatter(initial, final)
plt.plot([np.min(initial),np.max(initial)],[np.min(initial),np.max(initial)])
plt.title(f'{((initial-final)**2).mean(axis=None)}')
plt.savefig('results/scatter.pdf', transparent=True)
#-------------------------------------------------------------
