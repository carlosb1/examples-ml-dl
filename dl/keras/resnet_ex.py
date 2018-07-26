from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras
import tensorflow as tf
import numpy as np

model = ResNet50(weights='imagenet')
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print("Predicted: ", decode_predictions(preds, top=3)[0])

sess = keras.backend.get_session()
saver = tf.train.Saver()
save_path = saver.save(sess, '/home/carlosb/Desktop/save_models/resnet50_imagenet.ckpt')
