import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import time, random
import numpy as np
import core.utils as utils
import matplotlib.pyplot as plt
import tensorflow as tf
from core.configuration import cfg
from core.utils import read_class_names

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
from PIL import Image
from absl.flags import FLAGS
from core.yolo import filter_boxes
from absl import app, flags, logging
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

flags.DEFINE_string('framework', 'tf', '(tensorflow framework used')
flags.DEFINE_string('video_location', './test/video.mp4', 'path to input video and 0 for external webcam')
flags.DEFINE_string('weights_location', './weights/yolov4-416','path to tensorflow tf weights file')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'saving video to specific location')
flags.DEFINE_string('model', 'yolov4', 'yolo model')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('info', False, 'print information of detections cordinates on screen')



def main(_argv):
    
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_image_size = 416
    video_location = FLAGS.video_location
    

    saved_model_loaded = tf.saved_model.load(FLAGS.weights_location, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
        
    logging.info('weights loaded')
    logging.info('classes loaded')
    
    try:
        input_video = cv2.VideoCapture(int(video_location))
    except:
        input_video = cv2.VideoCapture(video_location)

    out = None

    if FLAGS.output:
        
        input_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        avg_fps = int(input_video.get(cv2.CAP_PROP_FPS))
        recording_codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, recording_codec, avg_fps, (input_width, input_height))

    while True:
        return_value, frame = input_video.read()
        
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            
        else:
            print('=> The Video has ended, try to input next video...')
            break
    
        frame_size = frame.shape[:2]
        video_image_data = cv2.resize(frame, (input_image_size, input_image_size))
        video_image_data = video_image_data / 255.
        video_image_data = video_image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(video_image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )

        """this will normalized ymin, xmin, ymax, xmax values of predicted bounding boxes"""
        
        pred_original_height, pred_original_width, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], pred_original_height , pred_original_width )
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        
        counted_classes = object_counting_function(pred_bbox,by_class = True) 

        for key, value in counted_classes.items():
                print("Number of {}s Detected: {}".format(key, value))
                #print("Total Number of Object Detected : {}".format(value))
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes)

        
        avg_fps = 1.0 / (time.time() - start_time)
        
        print("The Average FPS of Video: %.2f" % avg_fps ,"FPS")
        
        final_result = np.asarray(image)
        
        cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
        
        final_result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.putText(final_result, "Average FPS of Video: {:.2f}".format(avg_fps),(5,15),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(208,64,224),2 )
        
        cv2.imshow("Output", final_result)
        
        if FLAGS.output: 
            out.write(final_result)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("=> keyboard interrupt")
            break
        
    cv2.destroyAllWindows()


def object_counting_function(data, by_class = True):
    
    boxes, scores, classes, num_objects = data
    object_countings = dict()

    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        
        for i in range(num_objects):
            
            class_index = int(classes[i])
            class_name = class_names[class_index]
            object_countings[class_name] = object_countings.get(class_name, 0) + 1
        print("Total Number of Object Detected : {}".format(num_objects))
        

    else:
       print("detected Object Detected :")
         
    return object_countings

if __name__ == '__main__':
    
    try:
        app.run(main)
        
    except SystemExit:
        pass
