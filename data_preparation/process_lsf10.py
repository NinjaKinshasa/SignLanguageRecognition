import os
import cv2
import pandas as pd
import numpy as np

from os.path import join
from shutil import copyfile


def load_labels(input_folder):
    with open(join(input_folder, 'labels.csv'), "r") as f:
        labels = f.read().splitlines()
    return labels

def video_to_images(input_video_path, output_images_path, resize=None):

    os.makedirs(join(output_images_path), exist_ok=True)

    i = 0
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        print("ERROR : ", input_video_path)


    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if resize is not None:
                frame = cv2.resize(frame, dsize=resize, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_images_path, "{}.jpg".format(i)), frame)
        else:
            break
        i += 1

    cap.release()


def videos_to_images(input_folder, output_folder=None, resize=None, inplace=False):

    if inplace:
        output_folder = input_folder

    input_videos_path = join(input_folder, 'videos')
    output_videos_path = join(output_folder, 'videos')

    os.makedirs(output_videos_path, exist_ok=True)

    annotations = pd.read_csv(join(input_folder, 'annotations.csv'))

    for i, row in annotations.iterrows():

        video_id = row['video_id']
        input_video_path = os.path.join(input_videos_path, '{}.mp4'.format(video_id))
        output_images_path = os.path.join(output_videos_path, str(video_id))

        print('VIDEO TO IMAGES : ', input_video_path)
        video_to_images(input_video_path, output_images_path, resize)

        if inplace:
            os.remove(input_video_path)

    print("done")


def generate_annotations_classifier(input_folder, subset):

    print('GENERATE ANNOTATIONS : CLASSIFIER - ', subset)

    annotations_input_file = 'annotations.csv'
    annotations_output_file = 'annotations_classifier_{}.csv'.format(subset)

    annotations_output_path = join(input_folder, annotations_output_file)

    annotations_input = pd.read_csv(join(input_folder, annotations_input_file))

    f = open(join(input_folder, annotations_output_file), "w")
    f.write("video_id,label_id,total_frames,begin,end\n")

    for i, row in annotations_input.iterrows():

        end_frame = row['end']
        total_frames = row['total_frames']
        #print(begin_frame, end_frame, total_frames)

        if row['subset'] == subset:

            for shift in [0, 4, 8, 12]:

                begin_frame = row['begin'] + shift

                while begin_frame < end_frame:

                    #print("{},{},{},{},{}".format(row['video_id'], row['label_id'], total_frames, begin_frame, min(begin_frame + 15, end_frame)))
                    f.write("{},{},{},{},{}\n".format(row['video_id'], row['label_id'], total_frames, begin_frame, min(begin_frame + 15, end_frame)))
                    begin_frame += 16

    f.close()

    # align on 16 frames
    align_on_frames(annotations_output_path, align=16, inplace=True)

    # balance
    balance_csv(annotations_output_path, groupby='label_id', inplace=True)

    # shuffle
    shuffle_csv(annotations_output_path, inplace=True)



def generate_annotations_detector(input_folder):

    print('GENERATE ANNOTATIONS : DETECTOR')
    annotations_input_file = join(input_folder, 'annotations.csv')
    annotations_output_file = join(input_folder, 'annotations_detector.csv')

    labels = load_labels(input_folder)

    #generate annotations
    annotations_input = pd.read_csv(annotations_input_file)

    f = open(annotations_output_file, "w")
    f.write("video_id,is_gesture,total_frames,begin,end\n")

    no_gesture = 0
    gesture = 1


    for i, row in annotations_input.iterrows():

        gesture_begin = row['begin']
        gesture_end = row['end']
        total_frames = row['total_frames']
        #print(begin_frame, gesture_end, total_frames)

        if labels[row['label_id']] == 'pas_de_geste':

            for shift in [0, 3, 6]:
                window_begin = 0 + shift
                window_end = 7

                while window_end < total_frames:
                    #print("{},{},{},{},{}".format(row['video_id'], no_gesture, total_frames, begin_frame, gesture_end))
                    f.write("{},{},{},{},{}\n".format(row['video_id'], no_gesture, total_frames, window_begin, window_end))
                    window_begin += 8
                    window_end += 8


        if labels[row['label_id']] != 'pas_de_geste':

            # frames avant le geste
            window_begin = 0
            window_end = 7

            while window_end < gesture_begin:
                # print("{},{},{},{},{}".format(row['video_id'], no_gesture, total_frames, window_begin, window_end))
                f.write("{},{},{},{},{}\n".format(row['video_id'], no_gesture, total_frames, window_begin, window_end))
                window_begin += 8
                window_end += 8

            # frames pendant le geste
            for shift in [0, 4]:
                window_begin = gesture_begin + shift
                window_end = window_begin + 7

                while window_end < gesture_end:
                    # print("{},{},{},{},{}".format(row['video_id'], gesture, total_frames, begin_frame, min(begin_frame + 7, gesture_end)))
                    f.write("{},{},{},{},{}\n".format(row['video_id'], gesture, total_frames, window_begin, min(window_end, gesture_end)))

                    window_begin += 8
                    window_end += 8

            #frames après le geste
            window_end = gesture_end
            window_begin = window_end - 7

            while window_begin > gesture_end:
                # print("{},{},{},{},{}".format(row['video_id'], no_gesture, total_frames, gesture_end + 1, gesture_end + 8))
                f.write("{},{},{},{},{}\n".format(row['video_id'], no_gesture, total_frames, window_begin, window_end))
                window_begin -= 8
                window_end -= 8


    f.close()

    #align on 8 frames
    align_on_frames(annotations_output_file, 8, inplace=True)

    #shuffle
    shuffle_csv(annotations_output_file, inplace=True)

    #balance
    balance_csv(annotations_output_file, groupby='is_gesture', inplace=True)

    #split
    split_train_test(annotations_output_file, 0.85, inplace=True)

def align_on_frames(annotations_input_file, annotations_output_file=None, align=8, inplace=False):

    if inplace:
        annotations_output_file = annotations_input_file

    annotations_input = pd.read_csv(annotations_input_file)

    annotations_output = annotations_input.copy()
    padding_col = []

    for i, row in annotations_input.iterrows():

        padding = 0

        begin_frame = row['begin']
        end_frame = row['end']

        if end_frame < begin_frame + align - 1:
            padding = begin_frame + align - 1 - end_frame

        padding_col.append(padding)

    annotations_output['padding'] = padding_col
    annotations_output.to_csv(annotations_output_file, index=False)


def balance_csv(input_path, output_path=None, groupby='label_id', inplace=False):

    if inplace:
        output_path = input_path

    annotations = pd.read_csv(input_path)
    g = annotations.groupby(groupby)
    print('before balance : \n', g.count())
    balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))  # balance
    balanced.to_csv(output_path, index=False)

    balanced = pd.read_csv(output_path)
    g = balanced.groupby(groupby)
    print('\nafter balance : \n', g.count())


def shuffle_csv(input_path, output_path=None, inplace=False):

    if inplace:
        output_path = input_path

    annotations = pd.read_csv(input_path)
    suffled = annotations.sample(frac=1).reset_index(drop=True)  # shuffle
    suffled.to_csv(output_path, index=False)
    print('\nshuffle dataset')


def split_train_test(annotation_input_file, frac, inplace=False):

    annotations = pd.read_csv(annotation_input_file)
    np.random.seed(0)
    msk = np.random.rand(len(annotations)) < frac
    train = annotations[msk]
    val = annotations[~msk]

    path_no_ext = annotation_input_file[:-4]
    train.to_csv(path_no_ext + "_train.csv", index=False)
    val.to_csv(path_no_ext + "_val.csv", index=False)
    print()
    print('nb rows in train : ', len(train))
    print('nb rows in val : ', len(val))

    if inplace:
        os.remove(annotation_input_file)


def move_videos(input_folder, output_folder):

    annotations_input = pd.read_csv(join(input_folder, 'annotations.csv'))

    labels = load_labels(input_folder)
    copyfile(join(input_folder, 'labels.csv'), join(output_folder, 'labels.csv'))

    annotations_output = annotations_input.copy()
    annotations_output.columns = ['video_id','label_id','total_frames','begin','end', 'subset']

    for i, row in annotations_input.iterrows():
        video_id = str(i)

        old_video_name = row['video']
        new_video_name = video_id + '.mp4'

        label = row['label']
        label_id = str(labels.index(label))

        from_path = join(input_folder, 'videos', label, old_video_name)
        to_path = join(output_folder, 'videos', new_video_name)

        print('COPY : {} -> {}'.format(from_path, to_path))
        copyfile(from_path, to_path)

        annotations_output.at[i, 'video_id'] = video_id
        annotations_output.at[i, 'label_id'] = label_id

    annotations_output.to_csv(join(output_folder, 'annotations.csv'), index=False)

def process_dataset(input_folder, output_folder, resize):

    os.makedirs(join(output_folder, 'videos'), exist_ok=True)

    move_videos(input_folder=input_folder, output_folder=output_folder)
    videos_to_images(input_folder=output_folder, resize=resize, inplace=True)

if __name__ == "__main__":

    LSF10_raw = r'C:\Users\mdavid\Desktop\LSF10_raw'
    LSF10 = r'C:\Users\mdavid\Desktop\LSF10'

    process_dataset(input_folder=LSF10_raw, output_folder=LSF10, resize=(112, 112))

    generate_annotations_detector(input_folder=LSF10)
    generate_annotations_classifier(input_folder=LSF10, subset='train')
    generate_annotations_classifier(input_folder=LSF10, subset='val')
