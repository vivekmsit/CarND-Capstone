from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import zipfile
import tarfile
import os

import rospy

from urllib import urlopen
import six.moves.urllib as urllib

class TLClassifier(object):
    def __init__(self, download=False):
        #self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        #self.MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
        #self.MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

        # Download coco model
        if download is True:
            self.download_coco_model()

        # Path to frozen detection graph.
        PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'

        self.model = None
        self.width = 0
        self.height = 0
        self.channels = 3
        self.gamma = 0.6
        self.image_count = 0
        self.correct_gamma = True

        # Load a frozen model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            # Input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def download_coco_model(self):
        """Downloads coco model"""
        MODEL_FILE = self.MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        if os.path.exists(MODEL_FILE):
            print("model already downloaded")
            if os.path.exists(self.MODEL_NAME):
                return
        else:
            URL_PATH = DOWNLOAD_BASE + MODEL_FILE
            print("downloading model from " + URL_PATH)
            response = urlopen(URL_PATH)
            f = open(MODEL_FILE, 'wb')
            f.write(response.read())
            f.close()
            print("downloaded model successfully")
        print("extracting model")
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
                print("extracted model successfully")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.correct_gamma:
            if self.gamma == 1.0:
                self.gamma = 0.6
            elif self.gamma == 0.6:
                self.gamma = 1.0
        image = self.adjust_gamma(image, self.gamma)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.asarray(image, dtype="uint8")
        image_np_expanded = np.expand_dims(image_np, axis=0)

        detected = False

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        best_scores = []

        for idx, classID in enumerate(classes):
            if self.MODEL_NAME == 'ssdlite_mobilenet_v2_coco_2018_05_09':
                if classID == 10: # 10 is traffic light
                    if scores[idx] > 0.10: #confidence level
                        best_scores.append([scores[idx], idx, classID])
                        detected = True
            else: # we tuned the model to classify only traffic lights
                if scores[idx] > 0.10:  # confidence level
                    best_scores.append([scores[idx], idx, classID])
                    detected = True

        tl_index = -1
        if detected:
            best_scores.sort(key=lambda tup: tup[0], reverse=True)

            best_score = best_scores[0]
            rospy.loginfo("number of TL found %d, best score: %f, color: %f", len(best_scores), best_score[0], best_score[2])
            nbox = boxes[best_score[1]]

            height = image.shape[0]
            width = image.shape[1]

            box = np.array([nbox[0]*height, nbox[1]*width, nbox[2]*height, nbox[3]*width]).astype(int)
            box_height = box[2] - box[0]
            box_width = box[3] - box[1]
            ratio = float(box_height)/float(box_width)
            rospy.loginfo("ratio: %f", ratio)
            if ratio >= 2.0 and ratio < 3.0: #started from 2.4
                tl_cropped = image[box[0]:box[2], box[1]:box[3]]
                tl_color, tl_index = self.get_color(tl_cropped)
                rospy.loginfo("color is: %s", tl_color)
                #augment image with detected TLs
                cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (255, 255, 255)
                cv2.putText(image, tl_color, (box[1], box[0]), font, 2.0, font_color, lineType=cv2.LINE_AA)
        return image, tl_index

    def get_color(self, image_rgb):
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l = image_lab.copy()
        # set a and b channels to 0
        l[:, :, 1] = 0
        l[:, :, 2] = 0

        std_l = self.standardize_input(l)

        red_slice, yellow_slice, green_slice = self.slice_image(std_l)

        y, x, c = red_slice.shape
        px_sums = []
        color = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']
        px_sums.append(np.sum(red_slice[0:y, 0:x, 0]))
        px_sums.append(np.sum(yellow_slice[0:y, 0:x, 0]))
        px_sums.append(np.sum(green_slice[0:y, 0:x, 0]))

        max_value = max(px_sums)
        max_index = px_sums.index(max_value)

        return color[max_index], max_index

    def crop(self, image):
        row = 2
        col = 6
        cropped_img = image.copy()
        cropped_img = cropped_img[row:-row, col:-col, :]
        return cropped_img

    def standardize_input(self, image):
        standard_img = np.copy(image)
        standard_img = cv2.resize(standard_img, (32, 32))
        standard_img = self.crop(standard_img)
        return standard_img

    def slice_image(self, image):
        img = image.copy()
        shape = img.shape
        slice_height = int(shape[0] / 3)
        upper = img[0:slice_height, :, :]
        middle = img[slice_height:2 * slice_height, :, :]
        lower = img[2 * slice_height:3 * slice_height, :, :]
        return upper, middle, lower

    def enhance_image(self, image_bgr, gridsize=2): # Contrast Limited Adaptive Histogram Equalization
        image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        image_lab = cv2.merge(lab_planes)
        image_bgr = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
        image_bgr = cv2.GaussianBlur(image_bgr, (3, 3), 0)
        return image_bgr

    def adjust_gamma(self, bgr, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(bgr, table)
