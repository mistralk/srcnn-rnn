import numpy as np
import imageio
import pathlib
import os
from PIL import Image

from framework.util import load_images
from framework.model import reuse_model

if __name__ == '__main__':

    root_path = 'SR_dataset/Set5'
    image_paths = load_images(root_path)

    for image_path in image_paths:
        ground_truth = Image.open(image_path).convert('L')
        w, h = ground_truth.size
        downsampled = ground_truth.resize((w//2, h//2), Image.NEAREST)
        lowres = downsampled.resize((w, h), Image.BICUBIC)
        
        lowres = np.array(lowres)
        ground_truth = np.array(ground_truth)

        #for idx in range(1, 6):
        #    conv1_kernel = np.load('conv' + str(idx) + '_kernel.npy')
        #    np.load('conv' + str(idx) + '_bias.npy')
        #conv1_kernel = np.load('conv' + str(1) + '_kernel.npy')

        recovered = reuse_model(lowres, ground_truth, 'tmp/model.ckpt')
        recovered = recovered.astype(np.uint8)

        _, path_and_file = os.path.splitdrive(image_path)
        path, name = os.path.split(path_and_file)
        name = os.path.splitext(name)[0]

        imageio.imwrite(name + '_sr.png', recovered)