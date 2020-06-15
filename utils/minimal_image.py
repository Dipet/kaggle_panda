import cv2 as cv
import numpy as np


def __rotate_image(mat, rotation_mat):
    pass


def rotate_image(mat, angle, rect=None, default_val=255):
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
    try:
        rotated_mat = cv.warpAffine(
            mat, rotation_mat, (bound_w, bound_h), borderMode=cv.BORDER_CONSTANT, borderValue=(default_val, default_val, default_val)
        )
    except cv.error:
        # very large for warpAffine
        rotated_mat = np.empty([bound_h, bound_w, 3], dtype=np.uint8)

        # tl

    if rect is not None:
        (x, y), wh, a = rect
        xy = np.array([x, y, 1]) @ rotation_mat.T
        rect = tuple(xy), wh, 0
        return rotated_mat, rect

    return rotated_mat


def rect2points(rect, image_shape):
    box = cv.boxPoints(rect)
    box = np.int0(box)

    tl = box.min(axis=0).clip(0)
    br = box.max(axis=0).clip([0, 0], (image_shape[1], image_shape[0]))

    return tl, br


def get_sub_image(image, rect, filter_by_size=True, default_val=255):
    (x1, y1), (x2, y2) = rect2points(rect, image.shape)

    sub_image = image[y1:y2, x1:x2]
    if np.prod(sub_image.shape[:2]) < 10:
        return None

    (x, y), wh, a = rect
    if a == 0:
        s = (sub_image != 255).sum()
        total = np.prod(sub_image.shape[:2])
        if filter_by_size and s < 0.25 * total:
            return None

        h, w = sub_image.shape[:2]
        if h > w:
            sub_image = np.rot90(sub_image)
        return sub_image

    xy = (x - x1), (y - y1)
    rect = xy, wh, a
    sub_image, rect = rotate_image(sub_image, a, rect, default_val=default_val)

    return get_sub_image(sub_image, rect, filter_by_size)


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

        new_rectangles.append(box)

    offset = 0
    result_img = np.full(result_shape + [3], 255, dtype=np.uint8)
    for sub_img in result_sub_images:
        h, w = sub_img.shape[:2]
        result_img[offset : offset + h, :w] = sub_img
        offset += h

    return result_img, minimal_boxes


def get_compact(image, compact_representation, default_val=255):
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

    result_image = np.full(list(result_shape) + [3], default_val, dtype=np.uint8)

    offset = 0
    for box in boxes:
        (x, y), (w, h), a = box
        rect = (x * scale_w, y * scale_h), (w * scale_w, h * scale_h), a
        sub_image = get_sub_image(image, rect, filter_by_size=False, default_val=default_val)
        result_image[offset : offset + sub_image.shape[0], : sub_image.shape[1]] = sub_image
        offset += sub_image.shape[0]

    return result_image


if __name__ == "__main__":
    import os
    import json
    import skimage.io
    import matplotlib.pyplot as plt

    name = "00928370e2dfeb8a507667ef1d4efcbb"

    TRAIN = "../input/prostate-cancer-grade-assessment/train_images/"

    with open("../input/compact_representation.json", "r") as file:
        compact_representation = json.load(file)

    key = name + ".tiff"
    img = skimage.io.MultiImage(os.path.join(TRAIN, name + ".tiff"))[0]

    img = get_compact(img, compact_representation[key])

    plt.imshow(img)
    plt.show()
