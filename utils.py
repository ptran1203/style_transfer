import pickle
import numpy as np
import keras.preprocessing.image as image_processing

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