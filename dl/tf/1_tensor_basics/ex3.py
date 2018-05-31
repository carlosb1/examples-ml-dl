import tensorflow as tf

a = tf.get_variable(name="A", initializer=tf.constant(2))
b = tf.get_variable(name="B", initializer=tf.constant(3))
c = tf.add(a, b, name="Add")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

