import os
import json
import sys
sys.path.append("../")

from tqdm import tqdm
from skimage.io import MultiImage
import cv2 as cv

from utils.data_utils import get_tile
import matplotlib.pyplot as plt

images_dir = "../input/prostate-cancer-grade-assessment/train_images"
output_dir = "../input/256_36_hsv"

with open("../notebooks/256_36_hsv.json", 'r') as file:
    data = json.load(file)

os.makedirs(output_dir, exist_ok=True)


for path, boxes in tqdm(data.items()):
    img = MultiImage(os.path.join(images_dir, path) + ".tiff")[1]
    img = get_tile(img, boxes, 256, 36)
    img = 255 - cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(output_dir, path) + ".png", img)
