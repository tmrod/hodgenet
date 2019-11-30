#!/usr/bin/python3

import numpy as np
from scipy.linalg import null_space
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
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

np.random.seed(1776)
tf.set_random_seed(1776)
#--------------------------------------------

# user-supplied flags ---------------------------------------------
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('mask', 0.2, 'Mask percentage')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train')
flags.DEFINE_integer('iters_per_epoch', 10, 'Number of steps per epoch')
flags.DEFINE_integer('layers', 5, 'Number of RNN layers')
#flags.DEFINE_integer('early_stopping', 10, 'Early stopping count')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate')
flags.DEFINE_string('shift', 'hodge', 'Shift operator')
# hodge, linegraph, random, identity, unsigned
#------------------------------------------------------------------

# data loading ------------------------
f_file = 'flow.npy'
l_file = 'laplacian.npy'

#f_np = np.load(f_file)
#f_np /= np.mean(f_np)
#f_std = np.std(f_np)
#E = len(f_np)
#f_np = np.diag(f_np).astype(np.float32)
#f_np = np.array([f_np]).T.astype(np.float32)

L1_np = np.load(l_file)
E = L1_np.shape[0]
D = np.sum(np.abs(L1_np), axis=1) - 2
Dinv = np.diag(1/D)
D = np.diag(D)
sqrtDinv = np.sqrt(Dinv)

# synthetic flow data - cyclic + some white noise
flow_kernel = null_space(L1_np)
cyclic_flow = flow_kernel @ (np.random.randn(flow_kernel.shape[1])+1)
f_np = cyclic_flow + 0.1*np.random.randn(E)
f_np = np.array([f_np]).T.astype(np.float32)

#f_np = np.random.randn(E) # purely gradient flow
#f_np = L1_np @ f_np
#f_np = np.array([f_np]).T.astype(np.float32)


if(FLAGS.shift == 'hodge'):
    shift_np = L1_np
    shift_np *= 2/np.abs(np.max(np.linalg.eigvals(shift_np)))
elif(FLAGS.shift == 'random'):
    shift_np = np.random.rand(E, E)
    shift_np = (shift_np + shift_np.T)/2
    shift_np[shift_np < 0.8] = 0
    shift_np[shift_np > 0.7] = 1
    np.fill_diagonal(shift_np, 1.)
    shift_np *= 2/np.abs(np.max(np.linalg.eigvals(shift_np)))
elif(FLAGS.shift == 'linegraph'):
    shift_np = L1_np
    np.fill_diagonal(shift_np, 0)
    shift_np += D
    shift_np *= 2/np.abs(np.max(np.linalg.eigvals(shift_np)))
elif(FLAGS.shift == 'unsigned'):
    shift_np = -np.abs(L1_np)
    np.fill_diagonal(shift_np, 0)
    shift_np += D
    shift_np *= 2/np.abs(np.max(np.linalg.eigvals(shift_np)))
elif(FLAGS.shift == 'identity'):
    shift_np = np.identity(E)
else:
    print('Invalid parameter: shift')
    sys.exit()
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
                    shape=(1,dim),
                    initializer=weight_init)
V = tf.get_variable('V',
                    shape=(dim,dim),
                    initializer=weight_init)
W = tf.get_variable('W',
                    shape=(dim,1),
                    initializer=weight_init)
bh = tf.get_variable('bh',
                     dtype=tf.float32,
                     initializer=1.)
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
ignore_tf = tf.placeholder(tf.float32)
loss = tf.losses.mean_squared_error(output_layer*(1-mask_tf)*ignore_tf,
                                    ground_tf*(1-mask_tf)*ignore_tf) \
                                    /tf.reduce_mean((1-mask_tf)*ignore_tf)
loss_test = tf.losses.mean_squared_error(output_layer*(1-ignore_tf),
                                         ground_tf*(1-ignore_tf)) \
                                         /tf.reduce_mean(1-ignore_tf)
train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate). \
    minimize(loss)
#----------------------------------------------------------------------

# data processing functions ------------------------------
# returns corrupted data, as well as mask
def corrupt(x, FLAGS):
    mask = np.random.rand(x.shape[0],1)
    mask = (mask > FLAGS.mask).astype(np.float32)
    return mask*x, mask
# collapses diagonal matrix into column vector
def collapse(x):
    return np.array([np.diagonal(x)]).T.astype(np.float32)
#---------------------------------------------------------

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
    input_, mask_ = corrupt(observed_f_np, FLAGS)
    #ground_ = collapse(observed_f_np)
    ground_ = observed_f_np
    for _ in range(FLAGS.iters_per_epoch):
        sess.run(train_op, feed_dict={input_tf: input_,
                                      ground_tf: ground_,
                                      mask_tf: mask_,
                                      ignore_tf: observed_mask_np})
    cost = sess.run(loss, feed_dict={input_tf: input_,
                                     ground_tf: ground_,
                                     mask_tf: mask_,
                                     ignore_tf: observed_mask_np})
    costs.append(cost)
    cost_epochs.append(e)
    print('Epoch\t{:d}\tCost\t{:f}'.format(e, cost))
    if (e%10 == 0):
        test = sess.run(loss_test, feed_dict={input_tf: observed_f_np,
                                              ground_tf: true_f_np,
                                              ignore_tf: observed_mask_np})
        tests.append(test)
        test_epochs.append(e)
#-------------------------------------------------------------

# Plotting ---------------------------------------------------
plt.plot(cost_epochs, costs, label='Training')
plt.plot(test_epochs, tests, label='Testing')
plt.legend()
plt.title('{:s} shift operator'.format(FLAGS.shift))
plt.savefig('results/{:s}_training.png'.format(FLAGS.shift))
plt.gcf().clear()

mmm = observed_mask_np.T
mmm = 1-mmm[0]
mmm = mmm.astype(np.bool)

out0 = sess.run(o0, feed_dict={input_tf: observed_f_np})
out0 = np.array(out0)
out0 = out0[mmm]
out0 = out0.T[0]

out1 = sess.run(o1, feed_dict={input_tf: observed_f_np})
out1 = np.array(out1)
out1 = out1[mmm]
out1 = out1.T[0]

out2 = sess.run(o2, feed_dict={input_tf: observed_f_np})
out2 = np.array(out2)
out2 = out2[mmm]
out2 = out2.T[0]

out3 = sess.run(o3, feed_dict={input_tf: observed_f_np})
out3 = np.array(out3)
out3 = out3[mmm]
out3 = out3.T[0]

final = sess.run(output_layer, feed_dict={input_tf: observed_f_np})
final = np.array(final)
final = final[mmm]
final = final.T[0]

initial = np.copy(true_f_np)
initial = initial[mmm]
initial = initial.T[0]

plt.scatter(initial, out0, label='o0')
plt.scatter(initial, out1, label='o1')
plt.scatter(initial, out2, label='o2')
plt.scatter(initial, out3, label='o3')
plt.scatter(initial, final, label='final')
plt.legend()
plt.plot([np.min(initial),np.max(initial)],[np.min(initial),np.max(initial)])
plt.title('{:s} shift operator'.format(FLAGS.shift))
plt.savefig('results/{:s}_scatter.png'.format(FLAGS.shift))
#-------------------------------------------------------------
