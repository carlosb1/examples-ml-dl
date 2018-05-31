# imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))



# hyper-parameters
logs_path = "./logs/visActivation"  # path to the folder that we want to save the logs for Tensorboard
learning_rate = 0.001  # The optimization learning rate
epochs = 10  # Total number of training epochs
batch_size = 100  # Training batch size
display_freq = 100  # Frequency of displaying the training results

# Network Parameters
# We know that MNIST images are 28 pixels in each dimension.
img_h = img_w = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_h * img_w

# number of units in the hidden layer
h1 = 100

# level of the noise in noisy data
noise_level = 0.6

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)


def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    with tf.variable_scope(name):
        in_dim = x.get_shape()[1]
        W = weight_variable(name, shape=[in_dim, num_units])
        tf.summary.histogram('W', W)
        b = bias_variable(name, [num_units])
        tf.summary.histogram('b', b)
        layer = tf.matmul(x, W)
        layer += b
        if use_relu:
            layer = tf.nn.relu(layer)
        return layer, W


# Create graph
# Placeholders for inputs (x), outputs(y)
with tf.variable_scope('Input'):
    x_original = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_original')
    x_noisy = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X_noisy')


fc1, W1 = fc_layer(x_noisy, h1, 'Hidden_layer', use_relu=True)
out, W2 = fc_layer(fc1, img_size_flat, 'Output_layer', use_relu=False)

# calculate the activation 
h_active = W1 / tf.sqrt(tf.reduce_sum(tf.square(W1), axis=0)) # [784, 100]

# Define the loss function, optimizer, and accuracy
with tf.variable_scope('Train'):
    with tf.variable_scope('Loss'):
        loss = tf.reduce_mean(tf.losses.mean_squared_error(x_original, out), name='loss')
        tf.summary.scalar('loss', loss)
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Add 5 images from original, noisy and reconstructed samples to summaries
tf.summary.image('original', tf.reshape(x_original, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('noisy', tf.reshape(x_noisy, (-1, img_w, img_h, 1)), max_outputs=5)
tf.summary.image('reconstructed', tf.reshape(out, (-1, img_w, img_h, 1)), max_outputs=5)


# Merge all the summaries
merged = tf.summary.merge_all()


# Launch the graph (session)
sess = tf.InteractiveSession()
sess.run(init)
train_writer = tf.summary.FileWriter(logs_path, sess.graph)
num_tr_iter = int(mnist.train.num_examples / batch_size)
global_step = 0
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    for iteration in range(num_tr_iter):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x_noisy = batch_x + noise_level * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)

        global_step += 1
        # Run optimization op (backprop)
        feed_dict_batch = {x_original: batch_x, x_noisy: batch_x_noisy}
        _, summary_tr = sess.run([optimizer, merged], feed_dict=feed_dict_batch)
        train_writer.add_summary(summary_tr, global_step)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch = sess.run(loss,
                                  feed_dict=feed_dict_batch)
            print("iter {0:3d}:\t Reconstruction loss={1:.3f}".
                  format(iteration, loss_batch))

    # Run validation after every epoch
    x_valid_original  = mnist.validation.images
    x_valid_noisy = x_valid_original + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_valid_original.shape)

    feed_dict_valid = {x_original: x_valid_original, x_noisy: x_valid_noisy}
    loss_valid = sess.run(loss, feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.3f}".
          format(epoch + 1, loss_valid))
    print('---------------------------------------------------------')

def plot_max_active(x):
    """
    Plots the images that are maximally activating the hidden units
    :param x: numpy array of size [input_dim, num_hidden_units]
    """
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(17, 17))
    fig.subplots_adjust(hspace=.1, wspace=0)
    img_h = img_w = np.sqrt(x.shape[0]).astype(int)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(x[:, i].reshape((img_h, img_w)), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()

plot_max_active(sess.run(h_active))
# close the session after you are done with testing
sess.close()


