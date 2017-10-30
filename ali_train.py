#-*- coding:utf-8 -*-
import tensorflow as tf
import os
import sys
import argparse
from tensorflow.contrib import rnn

FLAGS = None


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """

  var = tf.get_variable(
      name,
      shape,
      initializer=tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def batch_norm(x, n_out):
    beta = tf.get_variable('beta', [n_out],initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable('gamma', [n_out], initializer=tf.constant_initializer(1.0))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return normed

def initialization(x,bshape):
    init = batch_norm(x,bshape)
    return init


def conv_step(x, wshape, bshape):
    kernel = _variable_with_weight_decay('weights', wshape, stddev=5e-2, wd=0.0)
    conv = tf.nn.conv2d(x, kernel , strides=[1, 1, 1, 1], padding='VALID')
    biases = tf.get_variable('biases', [bshape], initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv_bn = batch_norm(pre_activation, bshape)
    conv_re = tf.nn.relu(conv_bn)
    conv_pl = maxpool2d(conv_re, k=2)
    return conv_pl

def conv_allstep(x):
    with tf.variable_scope('init')  as scppe:
        init = initialization(x,4) 
    with tf.variable_scope('conv1') as scope:
        conv1 = conv_step(init, [2,2,4,8], 8)
    with tf.variable_scope('conv2') as scope:
        conv2 = conv_step(conv1, [3,3,8,16], 16)
    with tf.variable_scope('conv3') as scope:
        conv3 = conv_step(conv2, [3,3,16,32], 32)
    with tf.variable_scope('conv4') as scope:
        conv4 = conv_step(conv3, [2,2,32,64], 64)
        convall = tf.reshape(conv4,[-1,5*5*64])
    return convall

def conv_net(x):
    with tf.variable_scope('conv_net') as scope:
        x = tf.transpose(x, (1,0,2,3,4))
        out_list = []
        for i in range(x.shape[0]):
            out_list.append(conv_allstep(x[i]))
            scope.reuse_variables()
    return out_list


def RNN(x, n_hidden, bshape):

    with tf.variable_scope('rnn') as scope:
        #x = tf.unstack(x,15, 1)
        
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        weight = _variable_with_weight_decay('weights', [n_hidden, bshape], stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [bshape], initializer=tf.constant_initializer(0.1))
    return tf.matmul(outputs[-1], weight) + biases


def read_line(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    id, label, rader = tf.decode_csv(value, record_defaults=[['string'], ['string'],['string']],field_delim=",")
    rader = tf.string_split([rader]," ")
    rader = rader.values
    label = tf.string_to_number(label,"float32")
    rader = tf.string_to_number(rader,"float32")
    label = tf.reshape(label,[1])
    rader = tf.reshape(rader,[15,4,101,101])
    rader = tf.transpose(rader,[0,2,3,1])
    return label,rader
def read_line_batch(file_queue, batch_size):
    label, rader = read_line(file_queue)
    capacity = 3 * batch_size
    label_batch, rader_batch = tf.train.batch([label,rader], batch_size=batch_size, capacity=capacity,num_threads=10)
    return label_batch,rader_batch

def main(_):
    #训练数据集
    train_file_path = os.path.join(FLAGS.buckets, "remote-multipart2.txt")
    #模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")
    n_hidden = 256
    #训练集
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(train_file_path))
    train_labels, train_raders = read_line_batch(filename_queue,128)

    # calculate the prediction
    out = conv_net(train_raders)
    pred = RNN(out,n_hidden, 1)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.square(train_labels-pred))
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses')) 
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)
    
    # Initializing the variables
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。


    # Train and Test 
    
    for i in range(200):
        sess.run(optimizer)
        if ((i + 1) % 1 == 0):
            print("step:", i + 1, "accuracy:", sess.run(loss))

    print("accuracy: " , sess.run(loss))
    save_path = saver.save(sess, ckpt_path)
    print("Model saved in file: %s" % save_path) 
    
    coord.request_stop()
    coord.join(threads)
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)