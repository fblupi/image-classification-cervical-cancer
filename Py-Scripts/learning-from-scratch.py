import cv2
import glob
import numpy as np
import pandas as pd

from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing import Pool, cpu_count, freeze_support
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEP = '\\'

SEED = 14

RESIZE_TRAIN_IMAGES = False
RESIZE_TEST_IMAGES = False

TRAIN_IMAGES_FOLDER = 'train_extra_balanced_resized'
TEST_IMAGES_FOLDER = 'test_resized'
SIZE = 128

BATCH_SIZE = 15
NUM_EPOCHS = 10


def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0, 0]}]


def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (SIZE, SIZE), cv2.INTER_LINEAR)
    return [path, resized]


def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata


def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    if RESIZE_TRAIN_IMAGES:
        print('Reading train data from image files...')
        train = glob.glob('..' + SEP + 'data' + SEP + TRAIN_IMAGES_FOLDER + SEP + '**' + SEP + '*.jpg')
        train = pd.DataFrame([[p.split(SEP)[3], p.split(SEP)[4], p] for p in train], columns=['type', 'image', 'path'])
        train = im_stats(train)
        train = train[train['size'] != '0 0'].reset_index(drop=True)
        train_data = normalize_image_features(train['path'])
        np.save('.' + SEP + 'npy' + SEP + TRAIN_IMAGES_FOLDER + '-' + str(SIZE) + '.npy',
                train_data, allow_pickle=True, fix_imports=True)

        le = LabelEncoder()
        train_target = le.fit_transform(train['type'].values)
        np.save('.' + SEP + 'npy' + SEP + TRAIN_IMAGES_FOLDER + '-' + str(SIZE) + '-target.npy',
                train_target, allow_pickle=True, fix_imports=True)
    else:
        print('Reading train data from NPY files...')
        train_data = np.load('.' + SEP + 'npy' + SEP + TRAIN_IMAGES_FOLDER + '-' + str(SIZE) + '.npy',
                             allow_pickle=True, fix_imports=True)
        train_target = np.load('.' + SEP + 'npy' + SEP + TRAIN_IMAGES_FOLDER + '-' + str(SIZE) + '-target.npy',
                               allow_pickle=True, fix_imports=True)

    if RESIZE_TEST_IMAGES:
        print('Reading test data from image files...')
        test = glob.glob('..' + SEP + 'data' + SEP + TEST_IMAGES_FOLDER + SEP + '*.jpg')
        test = pd.DataFrame([[p.split(SEP)[3], p] for p in test], columns=['image', 'path'])
        test_data = normalize_image_features(test['path'])
        np.save('.' + SEP + 'npy' + SEP + TEST_IMAGES_FOLDER + '-' + str(SIZE) + '.npy',
                test_data, allow_pickle=True, fix_imports=True)

        test_id = test.image.values
        np.save('.' + SEP + 'npy' + SEP + TEST_IMAGES_FOLDER + '-' + str(SIZE) + '-label.npy',
                test_id, allow_pickle=True, fix_imports=True)
    else:
        print('Reading test data from NPY files...')
        test_data = np.load('.' + SEP + 'npy' + SEP + TEST_IMAGES_FOLDER + '-' + str(SIZE) + '.npy',
                            allow_pickle=True, fix_imports=True)
        test_id = np.load('.' + SEP + 'npy' + SEP + TEST_IMAGES_FOLDER + '-' + str(SIZE) + '-label.npy',
                          allow_pickle=True, fix_imports=True)

    np.random.seed(SEED)  # for reproducibility

    print('Generating validation data...')
    x_train, x_val_train, y_train, y_val_train = train_test_split(train_data, train_target,
                                                                  test_size=0.1, random_state=SEED)

    print('Data augmentation...')
    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    datagen.fit(train_data)

    print('Training model...')
    model = create_model()
    model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
                        validation_data=(x_val_train, y_val_train),
                        verbose=2, epochs=NUM_EPOCHS, steps_per_epoch=len(x_train))

    print('Predicting...')
    pred = model.predict_proba(test_data)

    print('Exporting to CSV...')
    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    freeze_support()
    main()
