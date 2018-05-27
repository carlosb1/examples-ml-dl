import tensorflow as tf


a = tf.constant(2)
b = tf.constant(3)

c = tf.add(a, b, name='Sum')
with tf.Session() as sess:
    print(sess.run(c))
