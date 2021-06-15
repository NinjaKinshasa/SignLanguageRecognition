import pandas as pd
from os.path import join
from os import listdir, rename
import cv2

def get_labels(file):
    with open(file, "r") as f:
        return f.read().splitlines()

def get_n_frames(folder):
    with open(join(folder,  'n_frames'), "r") as f:
        return int(f.read())

def rename_all(data_folder):
    #print(listdir(data_folder))

    for folder in listdir(data_folder):
        print(folder)
        n_frames = get_n_frames(join(data_folder, folder))
        for i in range(n_frames):
            old = str(i+1).zfill(5) + '.jpg'
            new = str(i) + '.jpg'
            old_path = join(data_folder, folder, old)
            new_path = join(data_folder, folder, new)
            #print('{} -> {}'.format(old_path, new_path))

            rename(old_path, new_path)

def resize_all(dataset_folder):
    #print(listdir(data_folder))

    for folder in listdir(dataset_folder):
        print(folder)
        n_frames = get_n_frames(join(dataset_folder, folder))
        for i in range(n_frames):

            img_path = join(dataset_folder, folder, str(i) + '.jpg')
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(img_path, image)
            #print('{} -> {}'.format(old_path, new_path))


def generate_annotations_classifier(input_folder, subset):


    annotations_input_file = join(input_folder, 'jester-v1-{}.csv'.format(subset))
    annotations_output_file = 'jester_annotations_classifier_{}.csv'.format(subset)

    annotations_output_path = join(input_folder, annotations_output_file)

    annotations_input = pd.read_csv(annotations_input_file, names=['video_id','label'], header=None, sep=';')

    labels = get_labels(join(input_folder, 'jester-v1-labels.csv'))

    f = open(annotations_output_path, "w")
    f.write("video_id,label_id,total_frames,begin,end,padding\n")

    for i, row in annotations_input.iterrows():

        begin_window = 0
        end_window = 15

        total_frames = get_n_frames(join(input_folder, 'videos', str(row['video_id'])))


        while end_window < total_frames:

            print("{},{},{},{},{},{}".format(row['video_id'], labels.index(row['label']), total_frames, begin_window, end_window, 0))

            f.write("{},{},{},{},{},{}\n".format(row['video_id'], labels.index(row['label']), total_frames, begin_window, end_window, 0))

            begin_window += 16
            end_window += 16


        if total_frames % 16 != 0:
            padding = end_window - total_frames + 1
            print("{},{},{},{},{},{}".format(row['video_id'], labels.index(row['label']), total_frames, begin_window, total_frames - 1, padding))
            f.write("{},{},{},{},{},{}\n".format(row['video_id'], labels.index(row['label']), total_frames, begin_window, total_frames - 1, padding))


    f.close()

    # balance
    balance_csv(annotations_output_path, groupby='label_id', inplace=True)

    # shuffle
    shuffle_csv(annotations_output_path, inplace=True)


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

if __name__ == "__main__":

    jester = r'C:\Users\mdavid\Desktop\jester_dataset'

    #rename_all(r'C:\Users\mdavid\Desktop\jester_dataset\data')
    resize_all(r'C:\Users\mdavid\Desktop\jester_dataset\videos')
    #generate_annotations_classifier(jester, 'train')
    #generate_annotations_classifier(jester, 'validation')