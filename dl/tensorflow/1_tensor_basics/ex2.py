import tensorflow as tf


a = tf.constant(2, name='A')
b = tf.constant(3, name='B')

c = tf.add(a, b, name='Sum')
with tf.Session() as sess:
    print(sess.run(c))
