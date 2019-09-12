import os, cv2
import numpy as np


def checkdirctexist(dirct):
    if not os.path.exists(dirct):
        os.makedirs(dirct)


def get_image_for_save(img, W, H, pad=6):
    img = img.data[0].numpy()
    img = img * 255.
    img[img < 0] = 0
    img[img > 255.] = 255.
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    img = img[32*pad:H+32*pad, 32*pad:W+32*pad, :]
    return img

def get_image_for_test(image_name, pad=6):
    img = cv2.imread(image_name)
    img = img.astype(np.float32)
    H, W, C = img.shape
    Wk = W
    Hk = H
    if W % 32:
        Wk = W + (32 * pad - W % 32)
    if H % 32:
        Hk = H + (32 * pad - H % 32)
    img = np.pad(img, ((32*pad, Hk - H), (32*pad, Wk - W), (0, 0)), 'reflect')
    im_input = img / 255.0
    im_input = np.expand_dims(np.rollaxis(im_input, 2), axis=0)
    return im_input, W, H
