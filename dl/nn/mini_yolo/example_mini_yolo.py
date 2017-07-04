from mini_yolo import MiniYolo
import matplotlib.pyplot as plt
import cv2

def draw_box(boxes,im,crop_dim):
    imgcv = im
    [xmin,xmax] = crop_dim[0]
    [ymin,ymax] = crop_dim[1]
    for b in boxes:
        h, w, _ = imgcv.shape
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        left = int(left*(xmax-xmin)/w + xmin)
        right = int(right*(xmax-xmin)/w + xmin)
        top = int(top*(ymax-ymin)/h + ymin)
        bot = int(bot*(ymax-ymin)/h + ymin)

        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        thick = int((h + w) // 150)
        cv2.rectangle(imgcv, (left, top), (right, bot), (255,0,0), thick)

    return imgcv

miniyolo = MiniYolo()
miniyolo.load_model('./model/yolo-tiny.weights')
#print miniyolo. model.summary()

imagePath = './test_resources/test1.jpg'
image = plt.imread(imagePath)
boxes = miniyolo.predict(image)
cv2.imwrite('result.bmp',draw_box(boxes,plt.imread(imagePath),[[500,1280],[300,650]]))


