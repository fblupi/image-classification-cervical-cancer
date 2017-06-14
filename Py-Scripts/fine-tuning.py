import cv2
import glob
import numpy as np
import pandas as pd

from datetime import datetime
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
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
SIZE = 224

BATCH_SIZE = 48
NUM_EPOCHS = 150

LOAD_MODEL = True
MODEL_PATH = '14-06-2017_09-40-epoch36-val_loss0.8113'


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
                                                                  test_size=0.25, random_state=SEED)

    print('Data augmentation...')
    datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
    datagen.fit(train_data)

    print('Creating model...')
    if LOAD_MODEL:
        model = load_model('.' + SEP + 'weights' + SEP + 'fine-tuning' + SEP + MODEL_PATH + '.hdf5')
    else:
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=None, input_shape=(SIZE, SIZE, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        output = Dense(3, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)
        for layer in model.layers[:len(model.layers) - 1]:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        print('First Training...')
        TIME_STAMP = datetime.today().strftime("%d-%m-%Y_%H-%M")
        checkpoint = ModelCheckpoint('.' + SEP + 'weights' + SEP + 'fine-tuning' + SEP + TIME_STAMP
                                     + "-epoch{epoch:02d}-val_loss{val_loss:.4f}" + '.hdf5',
                                     monitor='val_loss', save_best_only=False)
        model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
                            validation_data=(x_val_train, y_val_train), callbacks=[checkpoint],
                            verbose=1, epochs=20, steps_per_epoch=len(x_train) / BATCH_SIZE)

        for layer in model.layers[:100]:
            layer.trainable = False
        for layer in model.layers[100:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')

        print('Second training...')
        TIME_STAMP = datetime.today().strftime("%d-%m-%Y_%H-%M")
        checkpoint = ModelCheckpoint('.' + SEP + 'weights' + SEP + 'fine-tuning' + SEP + TIME_STAMP
                                     + "-epoch{epoch:02d}-val_loss{val_loss:.4f}" + '.hdf5',
                                     monitor='val_loss', save_best_only=False)
        model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
                            validation_data=(x_val_train, y_val_train), callbacks=[checkpoint],
                            verbose=1, epochs=NUM_EPOCHS, steps_per_epoch=len(x_train) / BATCH_SIZE)

    print('Predicting...')
    pred = model.predict(test_data)

    print('Exporting to CSV...')
    df = pd.DataFrame(pred, columns=['Type_1', 'Type_2', 'Type_3'])
    df['image_name'] = test_id
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    freeze_support()
    main()
