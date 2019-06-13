import argparse
import numpy as np
import imageio
import os
from PIL import Image

from framework.util import load_images
from framework.model import reuse_with_metric
from framework.model import reuse_without_metric

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='?', const=1, default='SR_dataset/Set5', type=str)
    args = ap.parse_args()
    root_path = args.input

    #root_path = 'SR_dataset/Set5'
    image_paths = load_images(root_path)

    for image_path in image_paths:
        ground_truth = Image.open(image_path).convert('L')
        w, h = ground_truth.size

        downsampled = ground_truth
        downsampled = ground_truth.resize((w//2, h//2), Image.BICUBIC)
        lowres = downsampled.resize((w, h), Image.BICUBIC)
        
        lowres = np.array(lowres)
        ground_truth = np.array(ground_truth)

        #recovered = reuse_without_metric(lowres, 'tmp/area5000/model.ckpt')    
        recovered = reuse_with_metric(lowres, ground_truth, 'tmp/model_a/model.ckpt')
        recovered = recovered.astype(np.uint8)

        _, path_and_file = os.path.splitdrive(image_path)
        path, name = os.path.split(path_and_file)
        name = os.path.splitext(name)[0]

        imageio.imwrite(name + '_sr.png', recovered)
        imageio.imwrite(name + '_lr.png', lowres)
        imageio.imwrite(name + '_gt.png', ground_truth)