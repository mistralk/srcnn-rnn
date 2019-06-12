import numpy as np
import tensorflow as tf
import imageio
import pathlib
from PIL import Image

from framework.util import load_images

if __name__ == '__main__':

    root_path = 'SR_dataset/Set5'
    image_paths = load_images(root_path)

    for image_path in image_paths:
        ground_truth = Image.open(image_path).convert('L')
        w, h = ground_truth.size
        downsampled = ground_truth.resize((w//2, h//2), Image.BICUBIC)
        lowres = downsampled.resize((w, h), Image.BICUBIC)

        #recovered
        

        imageio.imwrite(image_path + '_lr.png', lowres)
    