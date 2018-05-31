import tensorflow as tf

img_h = img_w = 28
img_size_flat = img_h * img_w
n_classes = 10
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("MNIST/",one_hot=True)

learning_rate = 0.001
batch_size= 100
num_steps = 100

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

W = tf.get_variable('W', dtype=tf.float32,
                    shape=[img_size_flat, n_classes],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
b = tf.get_variable('b',dtype=tf.float32,
                    initializer=tf.constant(0., shape=[n_classes]
                        , dtype=tf.float32))

output_logits = tf.matmul(x,W) + b
y_pred = tf.nn.softmax(output_logits)

#TODO check with is this
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits= output_logits),name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction= tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
saver = tf.train.Saver()


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(num_steps):
    x_batch, y_true_batch = data.train.next_batch(batch_size)

    feed_dict_train = {x: x_batch, y: y_true_batch}
    sess.run(optimizer,feed_dict= feed_dict_train)
    save_path = saver.save(sess,'./saved_variable')
    print('model saved in {}'.format(save_path))
