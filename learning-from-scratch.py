import pandas as pd
import glob
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

SIZE = 128
EPOCHS = 20
BATCH = 15


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
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata


def create_model(opt_):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, SIZE, SIZE)))
    model.add(Conv2D(8, (3, 3)))
    model.add(Dropout(0.2))
    model.add(Dense(12))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    train = glob.glob(".\\data\\train_extra_resized\\**\\*.jpg")
    train = pd.DataFrame([[p.split('\\')[3], p.split('\\')[4], p] for p in train], columns=['type', 'image', 'path'])
    test = glob.glob(".\\data\\test_resized\\*.jpg")
    test = pd.DataFrame([[p.split('\\')[3], p] for p in test], columns=['image', 'path'])

    train_data = normalize_image_features(train['path'])
    test_data = normalize_image_features(test['path'])

    print("Images have been successfully read")

    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)

    model = KerasClassifier(build_fn=create_model, nb_epoch=EPOCHS, batch_size=BATCH, verbose=2)

    opts_ = ['adamax']  # ['adadelta','sgd','adagrad','adam','adamax']
    epochs = np.array([EPOCHS])
    batches = np.array([BATCH])
    param_grid = dict(nb_epoch=epochs, batch_size=batches, opt_=opts_)
    grid = GridSearchCV(estimator=model, cv=StratifiedKFold(n_splits=2), param_grid=param_grid, verbose=20)
    grid_result = grid.fit(train_data, train_target)

    test_id = test.image.values

    pred = grid_result.predict_proba(test_data)
    df = pd.DataFrame(pred, columns=le.classes_)
    df['image_name'] = test_id
    df.to_csv('submission-keras.csv', index=False)


if __name__ == '__main__':
    freeze_support()
    main()