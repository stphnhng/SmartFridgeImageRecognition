#!/usr/bin/env python

import socket
import object_detection_app as obj_det

import time

import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import pymysql

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util




        
def connect(name,data):
    conn = pymysql.connect(host = 'sql3.freemysqlhosting.net', user = 'sql3197801', password = "dkqqPG6GXH", db = "sql3197801")


    try:
        a = conn.cursor()

        #sql = 'INSERT INTO  food, expired FROM eugeneTable;'
        script = 'UPDATE eugeneTable SET description=\'%s\' WHERE food=\'%s\';' % (data,  name)
        a.execute(script)
        conn.commit()
        print(script)
 
    except Error as error:
        print(error)


    # a.execute(sql)

    # countrow = a.execute(sql)
    
    # print("number of rows: ", countrow)
    
    # data = a.fetchall()

    # li = [[x[0],x[1]] for x in data]
    # # print(type(data[0][1]))
    # # print(data[0][1])

    # # # data = [list(temp) for a in temp]
    # # print(type(li))
    # # print(data)
    # return li


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
position = {}


def detect_objects(image_np, sess, detection_graph):
    global position
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    #for i in classes:
     #   print(i)
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    temp_boxes = np.squeeze(boxes);
    hasNonPerson = False
    for index,value in enumerate(classes[0]):
        if (scores[0,index] > 0.5 and category_index.get(value)['name'] != "person"):
            hasNonPerson = True
            print ('CATEGORY INDEX VALUE', category_index.get(value))
            print("-------------------------------" + str(temp_boxes[index, 0])+ str(temp_boxes[index, 1])+ str(temp_boxes[index, 2])+ str(temp_boxes[index, 3]))
            #position[category_index(value)['name']].append(temp_boxes[index, 3]) 
            if(category_index[value]['name'] in position):
                position[category_index[value]['name']].append([temp_boxes[index, 3]])
            else:
                position[category_index[value]['name']] = [temp_boxes[index, 3]]
    if(not hasNonPerson):
        position = {}

    for i in position:   
        if(position[i][0] < position[i][-1]):
            print(i)
            connect(i,"out")
            print("take out")
        else:
            print(i)
            connect(i,"in")
            print("take in")

    return image_np


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=480, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=360, help='Height of the frames in the video stream.')
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                    default=2, help='Number of workers.')
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                    default=5, help='Size of the queue.')
args = parser.parse_args()

logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)

input_q = Queue(maxsize=args.queue_size)
output_q = Queue(maxsize=args.queue_size)
pool = Pool(args.num_workers, worker, (input_q, output_q))
video_capture = WebcamVideoStream(src=args.video_source,
                                  width=args.width,
                                  height=args.height).start()
fps = FPS().start()



TCP_IP = '172.20.10.10'
TCP_PORT = 5050
BUFFER_SIZE = 1024
#MESSAGE = "Hello, World!"

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


isRunning = False

startTime = time.time()
while True:
    #s.send(MESSAGE)

    #run start video code
    frame = video_capture.read()
    input_q.put(frame)
    
    t = time.time()

    output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', output_rgb)
    fps.update()

    print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if(t - startTime >= 5.0):
        startTime = t
        data = str(s.recv(BUFFER_SIZE))
        print(data)
        if 'Stop' in data:
            #run terminate video code
            video_capture.stop()
        else:
            print('video capture should start now')
            video_capture = WebcamVideoStream(src=args.video_source,
                                  width=args.width,
                                  height=args.height).start()
        #print "received data:", data
    

s.close()
fps.stop()
print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

pool.terminate()
video_capture.stop()
cv2.destroyAllWindows()
