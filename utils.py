import pickle
import numpy as np
import urllib.request
import keras.preprocessing.image as image_processing
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

def pickle_save(object, path):
    try:
        print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))


def pickle_load(path):
    try:
        print("Loading data from {}".format(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
            print('load data successfully'.format(path))
            return data
    except Exception as e:
        print(str(e))
        return None

def norm(imgs):
    return (imgs - 127.5) / 127.5


def de_norm(imgs):
    return imgs * 127.5 + 127.5


def transform(x, seed=0):
    np.random.seed(seed)
    img = image_processing.random_rotation(x, 0.2)
    img = image_processing.random_shear(img, 30)
    img = image_processing.random_zoom(img, (0.5, 1.1))
    if np.random.rand() >= 0.5:
        img = np.fliplr(img)

    return img


def weighted_samples(labels, class_weight):
    w = []
    for i in range(len(labels)):
        w.append(class_weight[labels[i]])
    
    return np.array(w)

def show_images(img_array):
    shape = img_array.shape
    img_array = img_array.reshape(
        (-1, shape[-4], shape[-3], shape[-2], shape[-1])
    )
    # convert 1 channel to 3 channels
    channels = img_array.shape[-1]
    resolution = img_array.shape[2]
    img_rows = img_array.shape[0]
    img_cols = img_array.shape[1]

    img = np.full([resolution * img_rows, resolution * img_cols, channels], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[
            (resolution * r): (resolution * (r + 1)),
            (resolution * (c % 10)): (resolution * ((c % 10) + 1)),
            :] = img_array[r, c]

    img = (img * 127.5 + 127.5).astype(np.uint8)

    cv2_imshow(img)

def http_get_img(url, rst=64, gray=False, normalize=True):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.resize(img, (rst, rst))
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.reshape((1, rst, rst, -1))
    if normalize:
        img = norm(img)
    return img
