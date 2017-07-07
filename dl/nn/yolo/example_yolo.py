from keras.models import Sequential
from keras.layers import Reshape, Activation, Convolution2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense

from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD


import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2

import tensorflow as tf
wt_path = 'tiny-yolo-voc.weights'
#TODO add paths for datasets
ann_dir = './VOCdevkit/VOC2012/Annotations/'
img_dir = './VOCdevkit/VOC2012/JPEGImages/'


LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = [(43,206,72),(255,204,153),(128,128,128),(148,255,181),(143,124,0),(157,204,0),(194,0,136),(0,51,128),(255,164,5),(255,168,187),(66,102,0),(255,0,16),(94,241,242),(0,153,143),(224,255,102),(116,10,255),(153,0,0),(255,255,128),(255,255,0),(255,80,5)]



BATCH_SIZE = 8
BOX = 5
THRESHOLD = 0.2
ANCHORS = '1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]


NORM_H, NORM_W  = 416, 416
GRID_H, GRID_W = 13, 13
SCALE_NOOB, SCALE_CONF,SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
CLASS = 20


class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))
        
    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union
        
    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height
        
    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x1


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def interpret_netout(image, netout):
    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in xrange(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].probs[c] == 0: 
                continue
            else:
                for j in xrange(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        
        if max_prob > THRESHOLD:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])


            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), COLORS[max_indx], 2)
            cv2.putText(image, LABELS[max_indx], (xmin, ymin - 12), 0, 1e-3 * image.shape[0], (0,255,0), 2)
            
    return image

def parse_annotation(ann_dir):
    all_img = []
        
    for ann in os.listdir(ann_dir):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                all_img += [img]
                img['filename'] = elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        
                        if obj['name'] in LABELS:
                            img['object'] += [obj]
                        else:
                            break
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))
                        
    return all_img

def aug_img(train_instance):
    path = train_instance['filename']
    all_obj = copy.deepcopy(train_instance['object'][:])
    img = cv2.imread(img_dir + path)
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    t  = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t)

    img = img * (1 + t)
    img = img / (255. * 2.)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img[:,:,::-1]
    
    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(NORM_W) / w)
            obj[attr] = max(min(obj[attr], NORM_W), 0)
            
        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(NORM_H) / h)
            obj[attr] = max(min(obj[attr], NORM_H), 0)
            
        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = NORM_W - obj['xmax']
            obj['xmax'] = NORM_W - xmin
    
    return img, all_obj

def data_gen(all_img, batch_size):
    num_img = len(all_img)
    shuffled_indices = np.random.permutation(np.arange(num_img))
    l_bound = 0
    r_bound = batch_size if batch_size < num_img else num_img
    
    while True:
        if l_bound == r_bound:
            l_bound  = 0
            r_bound = batch_size if batch_size < num_img else num_img
            shuffled_indices = np.random.permutation(np.arange(num_img))
        
        batch_size = r_bound - l_bound
        currt_inst = 0
        x_batch = np.zeros((batch_size, NORM_W, NORM_H, 3))
        y_batch = np.zeros((batch_size, GRID_W, GRID_H, BOX, 5+CLASS))
        
        for index in shuffled_indices[l_bound:r_bound]:
            train_instance = all_img[index]
            
            # augment input image and fix object's position and size
            img, all_obj = aug_img(train_instance)
            #for obj in all_obj:
            #    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (1,1,0), 3)
            #plt.imshow(img); plt.show()
            
            # construct output from object's position and size
            for obj in all_obj:    
                box = []
                center_x = .5*(obj['xmin'] + obj['xmax']) #xmin, xmax
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = .5*(obj['ymin'] + obj['ymax']) #ymin, ymax
                center_y = center_y / (float(NORM_H) / GRID_H)
                
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                
                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_indx = LABELS.index(obj['name'])
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                    
                    y_batch[currt_inst, grid_y, grid_x, :, 0:4]        = BOX * [box]
                    y_batch[currt_inst, grid_y, grid_x, :, 4  ]        = BOX * [1.]
                    y_batch[currt_inst, grid_y, grid_x, :, 5: ]        = BOX * [[0.]*CLASS]
                    y_batch[currt_inst, grid_y, grid_x, :, 5+obj_indx] = 1.0
                
            # concatenate batch input from the image
            x_batch[currt_inst] = img
            currt_inst += 1
            
            del img, all_obj
        
        yield x_batch, y_batch
        
        l_bound  = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_img: r_bound = num_img

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def setup_model():
    model = Sequential()
    #layer 1
    model.add(Convolution2D(16,3,3, subsample=(1,1), border_mode='same', bias=False
            ,input_shape=(416,416,3)))

    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #layer 2 - 5
    for i in range(0,4):
        model.add(Convolution2D(32*(2**i), 3,3, subsample=(1,1), border_mode='same', bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    #layer 6
    model.add(Convolution2D(512,3,3, subsample=(1,1), border_mode='same', bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='same'))
    
    #layer 7-8 
    for _ in range (0,2):
        model.add(Convolution2D(1024, 3,3, subsample=(1,1), border_mode='same', bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Convolution2D(125,1,1,subsample=(1,1)))
    model.add(Activation('linear'))
    model.add(Reshape((13,13,5,25)))

    return model



weight_reader = WeightReader(wt_path)
model = setup_model()


print model.summary()


weight_reader.reset()

for i in range(len(model.layers)):
    if 'conv' in model.layers[i].name:
        if 'batch' in model.layers[i+1].name:
            norm_layer = model.layers[i+1]
            size = np.prod(norm_layer.get_weights()[0].shape)
            
            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)
            
            weights = norm_layer.set_weights([gamma, beta, mean, var])
            
        conv_layer = model.layers[i]
        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])



all_img = parse_annotation(ann_dir)

def custom_loss(y_true, y_pred):
    ### Adjust prediction
    # adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    
    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    
    ### Adjust ground truth
    # adjust x and y
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W)/GRID_W), (float(NORM_H)/GRID_H)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)
    
    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    
    # adjust confidence
    true_box_prob = y_true[:,:,:,:,5:]
    
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
    
    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor
    
    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf
    
    weight_prob = tf.concat(CLASS * [true_box_conf], 4) 
    weight_prob = SCALE_PROB * weight_prob 
    
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    
    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    
    return loss

layer = model.layers[-3] # the last convolutional layer
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

#tb_counter  = max([int(num) for num in os.listdir('./logs/')]) + 1
tensorboard = TensorBoard(log_dir='./logs/' + str(0), histogram_freq=0, write_graph=True, write_images=False)

sgd = SGD(lr=0.00001, decay=0.0005, momentum=0.9)

model.compile(loss=custom_loss, optimizer=sgd)#'adagrad')

print str(len(all_img) / BATCH_SIZE)

model.fit_generator(data_gen(all_img, BATCH_SIZE),
        int(len(all_img)/BATCH_SIZE),
        nb_epoch = 100,  
        verbose = 2, 
        callbacks = [early_stop, checkpoint, tensorboard], 
        max_q_size = 3)




#model.load_weights("weights.hdf5")
#image = cv2.imread('images/dog.jpg')

#plt.figure(figsize=(10,10))

#input_image = cv2.resize(image, (416, 416))
#input_image = input_image / 255.
#input_image = input_image[:,:,::-1]
#input_image = np.expand_dims(input_image, 0)

#netout = model.predict(input_image)

#print netout
#image = interpret_netout(image, netout[0])
#plt.imshow(image[:,:,::-1]); plt.show()

######### TODO VIDEOS 
#model.load_weights("weights.hdf5")

#video_inp = '../basic-yolo-keras/images/phnom_penh.mp4'
#video_out = '../basic-yolo-keras/images/phnom_penh_bbox.mp4'

#video_reader = cv2.VideoCapture(video_inp)
#video_writer = None

#while(True):
#    ret, frame = video_reader.read()
#
#    if ret == True:
#        input_image = cv2.resize(frame, (416, 416))
#        input_image = input_image / 255.
#        input_image = input_image[:,:,::-1]
#        input_image = np.expand_dims(input_image, 0)
#
#        netout = model.predict(input_image)
#
#        frame = interpret_netout(frame, netout[0])
#        
#        if video_writer is None:
#            fourcc = cv2.VideoWriter_fourcc(*'XVID')
#            video_writer = cv2.VideoWriter(video_out, fourcc, 50.0, (frame.shape[1], frame.shape[0]))
#            
#        video_writer.write(np.uint8(frame))
#        
#    else:
#        break
#
#video_reader.release()
#video_writer.release()


print "Loaded model"




