import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[784, 200], initializer=weight_initer)
bias_initer = tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

x_w = tf.matmul(X,W, name="MatMul")
x_w_b = tf.add(x_w, b, name="Add")
h = tf.nn.relu(x_w_b, name="ReLU")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    d = {X: np.random.rand(100,784)}
    print(sess.run(h, feed_dict=d))


