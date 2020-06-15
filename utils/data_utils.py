import numpy as np


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
