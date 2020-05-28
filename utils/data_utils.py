import numpy as np


def tile(img, sz=128, N=16):
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

    img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], constant_values=255)

    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

    if len(img) < N:
        img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:N]

    img = img[idxs]
    s = int(np.sqrt(N))
    result = np.empty([sz * s, sz * s, 3], dtype=np.uint8)
    for i in range(len(img)):
        x = i % s
        y = i // s
        result[y * sz : (y + 1) * sz, x * sz : (x + 1) * sz] = img[i]

    return result
