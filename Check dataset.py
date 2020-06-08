#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import cv2 as cv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json

from tqdm.notebook import tqdm

from skimage.io import MultiImage

import tifffile as tiff


# In[2]:


IMAGES = "input/prostate-cancer-grade-assessment/train_images"
MASKS = "dataset/train_label_masks"

CSV = "dataset/train.csv"


# In[3]:


def rotate_image(mat, angle, rect=None):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(
        mat, rotation_mat, (bound_w, bound_h), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )

    if rect is not None:
        (x, y), wh, a = rect
        xy = np.array([x, y, 1]) @ rotation_mat.T
        rect = tuple(xy), wh, 0
        return rotated_mat, rect

    return rotated_mat


# In[4]:


def rect2points(rect, image_shape):
    box = cv.boxPoints(rect)
    box = np.int0(box)

    tl = box.min(axis=0).clip(0)
    br = box.max(axis=0).clip([0, 0], (image_shape[1], image_shape[0]))

    return tl, br


def get_sub_image(image, rect):
    (x1, y1), (x2, y2) = rect2points(rect, image.shape)

    sub_image = image[y1:y2, x1:x2]
    if np.prod(sub_image.shape[:2]) < 10:
        return None

    (x, y), wh, a = rect
    if a == 0:
        s = (sub_image != 255).sum()
        total = np.prod(sub_image.shape[:2])
        if s < 0.25 * total:
            return None

        h, w = sub_image.shape[:2]
        if h > w:
            sub_image = np.rot90(sub_image)
        return sub_image

    xy = (x - x1), (y - y1)
    rect = xy, wh, a
    sub_image, rect = rotate_image(sub_image, a, rect)

    return get_sub_image(sub_image, rect)


# In[5]:


def get_minimal_image(image):
    raw_image = image
    if image.ndim > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = (image != 255).astype(np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (10, 10)))

    countours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    angle = 0
    prev_area = 0
    min_rects = []
    for c in countours:
        rect = cv.minAreaRect(c)
        cur_area = np.prod(rect[1])
        if cur_area > prev_area:
            prev_area = cur_area
            angle = rect[-1]
        min_rects.append(rect)

    height, width = image.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    shift = np.array([bound_w, bound_h]) / 2

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    result_shape = [0, 0]

    minimal_boxes = []
    result_sub_images = []

    new_rectangles = []
    for rect in min_rects:
        (x, y), (w, h), a = rect
        w += 5
        h += 5

        sub_image = get_sub_image(raw_image, rect)
        if sub_image is None:
            continue

        minimal_boxes.append(rect)
        result_shape[0] += sub_image.shape[0]
        result_shape[1] = max(result_shape[1], sub_image.shape[1])
        result_sub_images.append(sub_image)

        xy = np.array([x, y, 1]) @ rotation_mat.T
        rect = ((xy[0], xy[1]), (w, h), a - angle)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        tl = box.min(axis=0).clip(0)
        br = box.max(axis=0).clip([0, 0], [bound_w, bound_h])
        box = np.stack([tl, br])

        h = br[1] - tl[1]
        w = br[0] - tl[0]

        new_rectangles.append(box)

    offset = 0
    result_img = np.full(result_shape + [3], 255, dtype=np.uint8)
    for sub_img in result_sub_images:
        h, w = sub_img.shape[:2]
        result_img[offset : offset + h, :w] = sub_img
        offset += h

    return result_img, minimal_boxes


# In[6]:


names = [name for name in os.listdir(IMAGES)]
compact_representation = {}

mean_ratio = 0

for name in tqdm(names):
    img_path = os.path.join(IMAGES, name)

    img = MultiImage(img_path)[-1]

    compact_image, minimal_boxes = get_minimal_image(img)
    compact_representation[name] = {"original_size": img.shape[:2], "rectangles": minimal_boxes}

    mean_ratio += np.prod(compact_image.shape[:2]) / np.prod(img.shape[:2])
print(f"Mean ratio: {mean_ratio / len(names)}")
#
#
# # In[7]:
#
#
# with open("../dataset/compact_representation.json", "w") as file:
#     json.dump(compact_representation, file)


# In[8]:


with open("dataset/compact_representation.json", "r") as file:
    compact_representation = json.load(file)


# In[9]:


def get_compact(image, compact_representation):
    current_shape = image.shape[:2]
    original_size = compact_representation["original_size"]

    scale_h = current_shape[0] / original_size[0]
    scale_w = current_shape[1] / original_size[1]

    boxes = compact_representation["rectangles"]

    result_shape = [0, 0]
    for (x, y), (w, h), a in boxes:
        w, h = (w * scale_w, h * scale_h)
        if h > w:
            w, h = h, w
        shape = int(np.ceil(h)), int(np.ceil(w))

        result_shape[0] += int(shape[0])
        result_shape[1] = max(result_shape[1], int(shape[1]))

    result_image = np.full(list(result_shape) + [3], 255, dtype=np.uint8)

    offset = 0
    for box in boxes:
        (x, y), (w, h), a = box
        rect = (x * scale_w, y * scale_h), (w * scale_w, h * scale_h), a
        sub_image = get_sub_image(image, rect)
        result_image[offset : offset + sub_image.shape[0], : sub_image.shape[1]] = sub_image
        offset += sub_image.shape[0]

    return result_image


# In[12]:


names = [name[:-10] for name in os.listdir(MASKS)]

for name in tqdm(names[1:2]):
    img_path = os.path.join(IMAGES, name + ".tiff")
    img = tiff.imread(img_path)

    img2 = get_compact(img, compact_representation[name])

    h, w = img.shape[:2]
    img = cv.resize(img, (w // 10, h // 10))

    h, w = img2.shape[:2]
    img2 = cv.resize(img2, (w // 10, h // 10))

    print(img.shape, img2.shape)

    plt.figure()

    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(img2)

    plt.show()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
