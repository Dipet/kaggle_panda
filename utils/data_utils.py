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
