import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load previsouly trained model
autoencoder = load_model('autoencoder.h5')

# Get encoder layer from trained model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

# Array in which we will store computed scores
scores = []

# In order to save time on computations we keep only 1000 query images from test dataset 
n_test_samples = 1000

# Each time we will score the first 10 retrieved images, then the first 50 etc...
n_train_samples = [10, 50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                   20000, 30000, 40000, 50000, 60000]


def test_model(n_test_samples, n_train_samples):
    # Compute features for training dataset
    learned_codes = encoder.predict(x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    
    # Compute features for query images
    test_codes = encoder.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
    
    # We keep only n_test_samples query images from test dataset
    indexes = np.arange(len(y_test))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]
    
    # Compute score
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, y_train, n_train_samples)

for n_train_sample in n_train_samples:
    test_model(n_test_samples, n_train_sample)
 
# Save the computed scores into a file
np.save('computed_data/scores', np.array(scores))