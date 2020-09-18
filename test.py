import numpy as np

def preprocess(imgs):
    """
    BGR -> RBG then subtract the mean
    """
    return imgs[...,[2,1,0]] - np.array([103.939, 116.779, 123.68])


def deprocess(imgs):
    return imgs[...,[0,1,2]] + np.array([103.939, 116.779, 123.68])


img = np.full((10, 10, 3), 255)

decoded = preprocess(img)
back = deprocess(decoded)

print(np.mean(np.abs(img - back)))