import pathlib
import numpy as np


def load_images(root_path):
    root_path = pathlib.Path(root_path)
    image_paths = list(root_path.glob('*'))
    image_paths = [str(path) for path in image_paths]

    n_images = len(image_paths)
    print(n_images, 'images imported from', root_path)

    return image_paths


def split_train_test(dataset_list, train_set_ratio):
    np.random.shuffle(dataset_list)
    train_length = int(len(dataset_list) * train_set_ratio)
    train_set = dataset_list[:train_length]
    test_set = dataset_list[train_length:]
    return train_set, test_set