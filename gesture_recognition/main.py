from os.path import join

import cv2
import numpy as np
from tensorflow import keras

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

MODELS_FOLDER = r'..\models\best'
DATASET = 'LSF10'

class VideoFlow:


    def __init__(self, source):
        self.cap = cv2.VideoCapture(0 if source is 'webcam' else source)
        self.frame = None

        # Check if camera opened successfully
        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")

    def close_video(self):
        # When everything done, release the video capture object
        self.cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def next_frame(self):

        ret, self.frame = self.cap.read()
        if ret == True:

            self.frame = cv2.resize(self.frame, dsize=(800, 600), interpolation=cv2.INTER_CUBIC)
            return cv2.resize(self.frame, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        else:
            return None

    def next_8_frames(self):
        frames = []
        for i in range(8):
            frame = self.next_frame()
            if frame is None:
                return None
            else:
                frames.append(frame)
        return np.array(frames)

    def display(self):
        cv2.imshow('Frame', self.frame)

    def video_is_open(self):
        return self.cap.isOpened()

    def press_q(self):
        return cv2.waitKey(25) & 0xFF == ord('q')

    def text1(self, str):
        cv2.putText(self.frame, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def text2(self, str):
        cv2.putText(self.frame, str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


class Detector():

    def __init__(self):

        if DATASET == 'Jester':
            self.model = keras.models.load_model(join(MODELS_FOLDER, 'jester_detector.h5'))

        if DATASET == 'LSF10':
            self.model = keras.models.load_model(join(MODELS_FOLDER, 'LSF10_detector.h5'))

        #Jester

        self.text = "No gesture"
        self.active_window_size = 0

    def detection(self, frames):

        #print(np.array([frames]).shape)
        detection = self.model.predict(np.array([frames]))

        is_gesture = detection[0][0] >= 0.5

        if is_gesture:
            self.text = "Gesture"
            self.active_window_size += 1
        else:
            self.text = "No gesture"
            self.active_window_size = 0


        return self.text

def load_labels(labels_path):
    with open(labels_path, "r") as f:
        labels = f.read().splitlines()
    return labels

class Classifier():
    def __init__(self):

        if DATASET == 'Jester':
            self.model = keras.models.load_model(join(MODELS_FOLDER, 'jester_classifier.h5'))
            self.labels = load_labels(r'C:\Users\mdavid\PycharmProjects\GestureRecognition\labels.csv')

        if DATASET == 'LSF10':
            self.model = keras.models.load_model(join(MODELS_FOLDER, 'LSF10_classifier.h5'))
            self.labels = ['arbre', 'aujourdhui', 'bonjour', 'escalier', 'jaune', 'livre', 'manger', 'moto',
                           'ordinateur', 'voiture']


        self.last_8_predictions = []
        self.text = "No class"
        self.window = []
        self.j = 0 #successive active states
        self.early_detection = False


    def extend_window(self, frames):

        self.window.extend(frames)

        if len(self.window) < 16:
            return self.text
        if self.early_detection:
            return self.text

        if len(self.window) > 16:
            self.window = self.window[-16:]
        self.j += 1


        if self.j == 1:
            print('----------')
            alpha = 0
        else:
            alpha = self.probs * (self.j - 1)

        self.probs = self.model.predict(np.array([self.window]))[0]
        #print(self.probs)
        print(self.labels[np.argsort(self.probs)[-1]])
        self.text = 'predicting...'
        weights = 1. / (1. + np.exp(-0.2 * (self.j - 9.)))

        meanprobs = (alpha + weights * self.probs) / self.j

        sorted_idx = np.argsort(meanprobs)

        (max1, max2) = meanprobs[sorted_idx][-2:]

        if max2 - max1 >= 0.5:

            self.early_detection = True
            self.text = self.labels[sorted_idx[-1]]
            if self.text == 'Doing other things':
                self.text = ''

    def close_window(self):

        if self.j != 0:
            self.window = []
            self.j = 0

            max_idx = np.argsort(self.probs)[-1]
            max_val = self.probs[max_idx]
            if self.early_detection is False and max_val >= 0.15:
                print('end of move and no prediction')
                self.text = 'unable to predict : most likely ' + self.labels[max_idx]
            self.early_detection = False

    def classification(self):
        return self.text


def real_time_video_analysis(source):

    flow = VideoFlow(source)
    detector = Detector()
    classifier = Classifier()

    while flow.video_is_open():

        window_8frames = flow.next_8_frames()

        if window_8frames is None:
            break

        detection = detector.detection(window_8frames)

        if detection == "Gesture":
            classifier.extend_window(window_8frames)
        else:
            classifier.close_window()

        classification = classifier.classification()

        flow.text1(detection)
        flow.text2(classification)

        flow.display()

        if flow.press_q():
            break

    flow.close_video()

if __name__ == "__main__":

    with tf.device("/cpu:0"):
        #real_time_video_analysis('webcam')
        #real_time_video_analysis('webcam2.mp4')

        if DATASET == 'Jester':
            real_time_video_analysis(r'C:\Users\mdavid\PycharmProjects\GestureRecognition\webcam2.mp4')

        if DATASET == 'LSF10':

            videos = [
                'WIN_20210414_16_36_45_Pro.mp4',
                'WIN_20210414_16_37_38_Pro.mp4',
                'WIN_20210414_16_38_47_Pro.mp4',
                'WIN_20210414_16_40_24_Pro.mp4',
                'WIN_20210414_16_41_10_Pro.mp4',
                'WIN_20210414_16_42_18_Pro.mp4',
                'WIN_20210414_16_45_08_Pro.mp4',
                'WIN_20210414_16_47_25_Pro.mp4',
                'WIN_20210414_16_48_17_Pro.mp4',
                'WIN_20210414_16_50_32_Pro.mp4',
                'WIN_20210414_16_51_45_Pro.mp4',
                'WIN_20210414_16_53_14_Pro.mp4'
            ]

            for video in videos:
                real_time_video_analysis(os.path.join(r'C:\Users\mdavid\Desktop\LSF10_raw\phrases', video))