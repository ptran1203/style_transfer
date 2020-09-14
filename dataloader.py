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
    def __init__(self, base_dir, batch_size, rst, max_size=500):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.x = utils.pickle_load(
            os.path.join(self.base_dir, 'dataset/content_imgs_{}.pkl'.format(rst)))[:max_size]

        self.y = utils.pickle_load(
            os.path.join(self.base_dir, 'dataset/style_imgs_{}.pkl'.format(rst)))[:max_size]

        self.x = utils.norm(self.x)
        self.y = utils.norm(self.y)
        # self.x, self.x_test, self.y, self.y_test = train_test_split(self.x, self.y,
        #                                                             test_size=0.2,
        #                                                             random_state=42)


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
        x = self.x
        y = self.shuffle_style_imgs()

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]

            yield self.augment_array(
                x[access_pattern, :, :, :],
                y[access_pattern],
                augment_factor,
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
