from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):

  # initialize parameters
  n_epochs = 10000
  batch_size = 100
  learning_rate = .5

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  usps_images = np.loadtxt("USPS/test_image.txt")
  usps_labels = np.loadtxt("USPS/test_label.txt")
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.truncated_normal([784, 10],stddev = 0.05))
  b = tf.Variable(tf.truncated_normal([10],stddev = 0.05))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # applying cross-entropy to evaluate correctness of prediction
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  # training model using gradient descent optimizer
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

  # initialize interactive session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train model
  for _ in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

  # calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  final_acc = sess.run(accuracy, feed_dict={x:  mnist.test.images,
                                      y_: mnist.test.labels})
  

  # save the results to file
  Stxt = (str(n_epochs)) + " " + str(batch_size) + " " + str(learning_rate) + " " + str(final_acc) + "\n";
  print(Stxt);
  F = open("MNLR.txt", "a");
  F.write(Stxt)
  F.close();

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    

