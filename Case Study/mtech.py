import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import core.utils as utils

import pytesseract
from core.utils import read_class_names
from core.config import cfg
from PIL import Image
from absl.flags import FLAGS
from absl import app, flags, logging
from core.yolov4 import filter_boxes
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string('model', 'yolov4', 'yolo model is used ')
flags.DEFINE_list('images', './car.jpg', 'path to input image')
flags.DEFINE_string('framework', 'tf', 'tensorflow framework is used')
flags.DEFINE_boolean('tiny', False, 'argument for yolo tiny model')
flags.DEFINE_string('output', './output/', 'location to output folder')


def main(_argv):
    
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_image_size = 416
    image_input_location = FLAGS.images

    saved_model_loaded = tf.saved_model.load('./weights/custom-416', tags=[tag_constants.SERVING])
    
    logging.info('weights loaded')
    logging.info('classes loaded')
    
    for count, image_location in enumerate(image_input_location, 1):
        
        original_image = cv2.imread(image_location)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_info_data = cv2.resize(original_image, (input_image_size, input_image_size))
        image_info_data = image_info_data / 255.
       
        image_name = image_location.split('/')[-1]
        image_name = image_name.split('.')[0]

        images_data = []
        
        for i in range(1):
            images_data.append(image_info_data)
        images_data = np.asarray(images_data).astype(np.float32)


        infer = saved_model_loaded.signatures['serving_default']
        batch_value = tf.constant(images_data)
        pred_bbox = infer(batch_value)
        
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold= 0.45,
            score_threshold=0.40
        )

        original_height, original_width, _ = original_image.shape
        
        bboxes = utils.format_boxes(boxes.numpy()[0], original_height,  original_width)
        
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        allowed_classes = list(class_names.values())


        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes, read_plate = True)
        
        image = Image.fromarray(image.astype(np.uint8))
        
        image.show()
        
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        
        class_index = int(classes[i])
        class_name = class_names[class_index]
        
        xmin, ymin, xmax, ymax = boxes[i]
        
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
        
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        blur = cv2.medianBlur(thresh, 3)
        
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None

if __name__ == '__main__':
    
    try:
        app.run(main)
        
    except SystemExit:
        pass
