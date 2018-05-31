import tensorflow as tf


s = tf.constant(2.3, name='scalar', dtype=tf.float32)
m = tf.constant([[1, 2], [3, 4]], name='matrix')

with tf.Session() as sess:
    print(sess.run(s))
    print(sess.run(m))
