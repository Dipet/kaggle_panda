import numpy as np
import torch
from torch.nn import functional
import cv2 as cv


def tile(img, sz=128, N=16, transform=None, random=False):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255)

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    if len(img) < N:
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]

    img = img[idxs]
    s = int(np.sqrt(N))
    result = np.full([sz * s, sz * s, 3], 255, dtype=np.uint8)

    indexes = np.arange(N)
    if random:
        np.random.shuffle(indexes)

    for i, j in enumerate(indexes):
        if i >= len(img):
            break
        x = j % s
        y = j // s
        img_i = img[i] if transform is None else transform(image=img[i])["image"]
        if result.dtype != img_i.dtype:
            result = result.astype(img_i.dtype)
        result[y * sz:(y + 1) * sz, x * sz:(x + 1) * sz] = img_i

    return result


def __rgb_to_hsv(image):
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (3, H, W). Got {}".format(image.shape))

    r, g, b = image

    maxc = image.max(-3)[0]
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac: torch.Tensor = maxc - minc

    s = deltac / v
    s = torch.where(torch.isnan(s), torch.zeros_like(s), s)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)

    maxg = g == maxc
    maxr = r == maxc

    r, g, b = (torch.stack([maxc] * 3) - image) * (1 / deltac)

    h = 4.0 + g - r
    h = torch.where(maxg, 2.0 + r - b, h)
    h = torch.where(maxr, b - g, h)
    h = torch.where(minc == maxc, torch.zeros_like(h), h)

    h *= 60.0
    h %= 360.0

    return torch.stack([h, s, v], dim=-3)


def get_tile(img, boxes, sz=256, num=36):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], constant_values=255)

    s = int(np.sqrt(num))
    tile = np.full([sz * s, sz * s, 3], 255, dtype=img.dtype)
    for i, box in enumerate(boxes):
        x = i % s
        y = i // s

        try:
            tile[y * sz:(y + 1) * sz, x * sz:(x + 1) * sz] = img[box[1]:box[3], box[0]:box[2]]
        except Exception:
            pass
    return tile


def tile_hsv_boxes(img, sz=256, N=36, default_val=255):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    num_channels = 1

    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)[..., 1]
    img = 255 - torch.from_numpy(img).cuda()
    img = torch.unsqueeze(img, -1)

    img = functional.pad(img, [0, 0, pad1 // 2, pad1 - pad1 // 2, pad0 // 2, pad0 - pad0 // 2], value=default_val)

    shape = img.shape

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, num_channels)
    img = img.permute([0, 2, 1, 3, 4]).reshape(-1, sz, sz, num_channels)

    num = len(img)

    if len(img) < N:
        img = functional.pad(img, [0, 0, 0, 0, 0, 0, 0, N - len(img)], value=default_val)

    idxs = torch.argsort(img.reshape(img.shape[0], -1).float().sum(-1))[:N]

    s = int(np.sqrt(N))
    res_boxes = []
    idxs = idxs.detach().cpu().numpy()

    indexes = np.arange(N)

    for i, j in enumerate(indexes):
        if i >= num:
            break

        x = idxs[i] % (shape[1] // sz)
        y = idxs[i] // (shape[1] // sz)
        x1 = x * sz
        x2 = (x + 1) * sz
        y1 = y * sz
        y2 = (y + 1) * sz
        res_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return res_boxes


def _find_bounding_boxes(image, preprocessed=False):
    if not preprocessed:
        if image.ndim > 2:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = (image != 255).astype(np.uint8)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (10, 10)))
    countours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    min_rects = []
    for c in countours:
        rect = cv.boundingRect(c)
        min_rects.append(rect)

    return [[x, y, x + w, y + h] for x, y, w, h in min_rects]


def moving_sum(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


def _find_new_tile_boxes(img, sz=64, threshold=0, max_overlap=0.0):
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = 255 - img

    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = cv.copyMakeBorder(img,
                            top=0, left=sz,
                            bottom=pad0 + sz, right=pad1 + sz,
                            borderType=cv.BORDER_CONSTANT, value=0)

    sums = []
    bboxes = []
    for i in range(0, img.shape[0], sz):
        y1 = i
        y2 = i + sz

        sub_img = img[y1:y2]
        m_sum = moving_sum(sub_img.sum(axis=0), sz)

        indexes = np.argsort(m_sum)

        cond = m_sum[indexes] > threshold
        indexes = indexes[cond]

        while len(indexes):
            x1 = indexes[-1]
            x2 = x1 + sz
            cond1 = indexes > (x1 - int(sz * (1 - max_overlap)))
            cond2 = indexes <= (x2 - int(sz * max_overlap))
            indexes = indexes[~(cond1 & cond2)]

            sums.append(int(m_sum[x1]))

            x1 -= sz
            x2 = x2 - sz
            bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes), np.array(sums)


def get_new_tile_boxes(img, size):
    h, w = img.shape[:2]

    boxes = _find_bounding_boxes(img)

    tile_boxes = []
    tile_sums = []
    for i, box in enumerate(boxes):
        sub_img = img[box[1]:box[3], box[0]:box[2]]

        b, s = _find_new_tile_boxes(sub_img, size, max_overlap=0.0)
        b = b.astype(np.float32)
        if not len(b):
            continue

        try:
            b[:, 0::2] = (b[:, 0::2] + box[0]) / w
            b[:, 1::2] = (b[:, 1::2] + box[1]) / h
        except Exception as err:
            print(b.shape, box)
            raise err
        tile_boxes.append(b)
        tile_sums.append(s)

    if len(tile_sums):
        tile_sums = np.concatenate(tile_sums)
        tile_boxes = np.concatenate(tile_boxes)
    else:
        tile_sums = np.empty([0])
        tile_boxes = np.empty([0, 4])

    return tile_boxes, tile_sums


def get_tiles_new(img, boxes, sums, size=64, num=36, pad_value=(255, 255, 255), random=False):
    h, w = img.shape[:2]

    s = int(np.sqrt(num))
    result = np.full([s * size, s * size, 3], pad_value, dtype=img.dtype)

    boxes = boxes[np.argsort(sums)[::-1]][:num]
    if random:
        np.random.shuffle(boxes)
    for i, box in enumerate(boxes):
        x = i % s
        y = i // s

        box = box.copy()
        box[0::2] *= w
        box[1::2] *= h
        x1, y1, x2, y2 = box.astype(int)

        tile = img[y1:y2, x1:x2]
        th, tw = tile.shape[:2]
        if th > size or tw > size:
            tile = tile[:size, :size]
        if th < size or tw < size:
            th, tw = tile.shape[:2]
            pad_h = size - th
            pad_w = size - tw
            tile = cv.copyMakeBorder(tile, top=0, left=0, bottom=pad_h, right=pad_w,
                                     borderType=cv.BORDER_CONSTANT, value=pad_value)

        result[y * size:(y + 1) * size, x * size:(x + 1) * size] = tile

    return result


def iafoss_tile_boxes_on_original_image(img, sz=256, N=100):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]],
                 constant_values=255)

    shape = img.shape

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    num = len(img)

    if len(img) < N:
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]

    img = img[idxs]
    s = int(np.sqrt(N))

    indexes = np.arange(N)

    boxes = []
    for i, j in enumerate(indexes):
        if i >= len(img):
            break
        if i >= num:
            break

        x = idxs[i] % (shape[1] // sz)
        y = idxs[i] // (shape[1] // sz)
        x1 = x * sz
        x2 = (x + 1) * sz
        y1 = y * sz
        y2 = (y + 1) * sz
        x1 -= pad1
        x2 -= pad1
        y1 -= pad0
        y2 -= pad0
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes)
