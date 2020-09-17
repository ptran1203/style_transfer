import numpy as np
import utils
from collections import Counter
import os
from sklearn.model_selection import train_test_split
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    from cv2 import imshow as cv2_imshow

class DataGenerator:
    def __init__(self, base_dir, batch_size, rst, max_size=500,
    multi_batch=False, normalize=True):
        BATCH_FILES = 4
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.id = 1
        self.rst = rst
        self.multi_batch = multi_batch
        self.x = utils.pickle_load(
            os.path.join(self.base_dir, 'dataset/content_imgs_{}.pkl'.format(rst)))[:max_size]

        if multi_batch:
            self.y = utils.pickle_load(
                os.path.join(self.base_dir, 'dataset/style_imgs_{}_{}.pkl'.format(rst, self.id)))[:max_size]
        else:
            self.y = utils.pickle_load(
                os.path.join(self.base_dir, 'dataset/style_imgs_{}.pkl'.format(rst)))[:max_size]

        self.max_size = max_size

        if normalize:
            self.x = utils.norm(self.x)
            self.y = utils.norm(self.y)


    def next_id(self):
        self.id += 1
        if self.id > self.BATCH_FILES:
            self.id = 1
        
        self.y = utils.pickle_load(
            os.path.join(
                self.base_dir, 'dataset/style_imgs_{}_{}.pkl'.format(self.rst, self.id))
        )[:self.max_size]
        self.y = utils.norm(self.y)


    def augment_one(self, x, y):
        seed = np.random.randint(0, 100)
        new_x = utils.transform(x, seed)
        new_y = utils.transform(y, seed)
        return new_x, new_y


    def augment_array(self, x, y, augment_factor):
        imgs = []
        masks = []
        for i in range(len(x)):
            imgs.append(x[i])
            masks.append(y[i])
            for _ in range(augment_factor):
                _x, _y = self.augment_one(x[i], y[i])
                imgs.append(_x)
                masks.append(_y)

        return np.array(imgs), np.array(masks)


    def shuffle_style_imgs(self):
        size = len(self.y)
        indices = np.arange(size)
        np.random.shuffle(indices)
        return self.y[indices]


    def next_batch(self, augment_factor):
        if self.multi_batch:
            x = self.x
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            max_id = x.shape[0] - self.batch_size + 1
            print("[", end="")
            for i in range(self.BATCH_FILES):
                for start_idx in range(0, max_id, self.batch_size):
                    access_pattern = indices[start_idx:start_idx + self.batch_size]

                    yield (
                        x[access_pattern, :, :, :],
                        self.y[access_pattern],
                    )
                print("{}/6 - ".format(i+1), end="")
                self.next_id()
            print("]")
        else:
            x = self.x
            self.y = self.shuffle_style_imgs()

            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            max_id = x.shape[0] - self.batch_size + 1
            for start_idx in range(0, max_id, self.batch_size):
                access_pattern = indices[start_idx:start_idx + self.batch_size]

                yield (
                    x[access_pattern, :, :, :],
                    self.y[access_pattern],
                )

    def get_random_sample(self, test=True):
        if test:
            idx = np.random.randint(0, self.x_test.shape - 1)
            return self.x_test[idx], self.y_test[idx]

        idx = np.random.randint(0, self.x.shape - 1)
        return self.x[idx], self.y[idx]


    def random_show(self, option='style'):
        """
        option: ['style', 'content']
        """
        idx = np.random.randint(0, self.x.shape - 1)
        if option == 'style':
            return cv2_imshow(utils.de_norm(self.y[idx]))

        return cv2_imshow(utils.de_norm(self.x[idx]))
