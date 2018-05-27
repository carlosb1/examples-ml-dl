import tensorflow as tf
weights = tf.get_variable(name="W", shape=[2,3], initializer=tf.truncated_normal_initializer(stddev=0.01))
biases = tf.get_variable(name="b", shape=[3], initializer=tf.zeros_initializer())

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    W,b = sess.run([weights, biases])
    print ('weights = {}'.format(W))
    print ('biases = {}'.format(b))



