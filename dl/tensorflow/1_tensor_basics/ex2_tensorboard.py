import tensorflow as tf
tf.reset_default_graph()


x_scalar = tf.get_variable('x_scalar', shape=[],initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    for step in range(100):
        sess.run(init)
        summary = sess.run(first_summary)
        writer.add_summary(summary, step)

    print('Done with writing the scalar summary')






