import tensorflow as tf
tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('saved_variable.meta')

with tf.Session() as sess:
    imported_graph.restore(sess, './saved_variable')
    weight, bias = sess.run(['W:0','b:0'])
    print('W = ', weight)
    print('b = ', bias)


