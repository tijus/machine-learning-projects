from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def main(_):

  # initialize parameters
  n_epochs = 20000
  batch_size = 500
  n_hidden = 256
  learning_rate = 5.5

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  usps_images = np.loadtxt("USPS/test_image.txt")
  usps_labels = np.loadtxt("USPS/test_label.txt")

  # create space for input image
  x = tf.placeholder(tf.float32, [None, 784])
  # create space for input labels
  y = tf.placeholder(tf.float32, [None, 10])

  # create weights and bias from input layer to hidden layer
  W1 = tf.Variable(tf.truncated_normal([784, n_hidden],stddev = 0.05))
  b1 = tf.Variable(tf.truncated_normal([n_hidden],stddev = 0.05))

  # create weights and bias from hidden layer to output layer
  W2 = tf.Variable(tf.truncated_normal([n_hidden, 10],stddev = 0.05))
  b2 = tf.Variable(tf.truncated_normal([10],stddev = 0.05))

  # output from the hidden layer with activation function as sigmoid
  h = tf.nn.relu(tf.add(tf.matmul(x,W1),b1))

  # applying softmax regression as activation function to output layer
  y_ = tf.nn.softmax(tf.add(tf.matmul(h,W2),b2))

  # Defining training step by reducing cross entropy
  cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

  # Applying gradient descent optimizer until cross_entropy is minimized
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

  # create interactive session
  sess = tf.InteractiveSession()

  # initialize global variable
  tf.global_variables_initializer().run()

  # Train model
  for _ in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  final_acc = (sess.run(accuracy, feed_dict={x:  usps_images,
                                      y: usps_labels}))

  # Store the results to file
  Stxt = (str(learning_rate) + " " + str(n_epochs)) + " " + str(batch_size) + " " + str(n_hidden) + " " + str(final_acc) + "\n";
  print(Stxt);
  F = open("MNN.txt", "a");
  F.write(Stxt)
  F.close();


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)