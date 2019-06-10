import os
from datetime import datetime

from framework.util import load_images
from framework.util import split_train_test
from framework.input import input_dataset
from framework.model import model
from framework.model import reuse_model
from framework.train import train_and_test

# Ignore warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_Set5():
    set5_paths = load_images('SR_dataset/Set5')
    set5_inputs = input_dataset(set5_paths, len(set5_paths))
    set5_spec = model(set5_inputs, training=False)
    reuse_model('tmp/model.ckpt', set5_spec)


if __name__ == '__main__':
    # TODO: command line interface
    
    n_epoch = 10
    batch_size = 128
    
    # Create two dataset (input data pipeline with image paths)
    image_paths = load_images('SR_dataset/291')
    train_paths, test_paths = split_train_test(image_paths, 0.8)

    # Timer start
    time_fmt = '%H:%M:%S'
    present = datetime.now().strftime(time_fmt)
    
    # Create two iterators over the two datasets
    train_inputs = input_dataset(train_paths, batch_size)
    test_inputs = input_dataset(test_paths, len(test_paths))

    # Define the model and save two model specifications for train and test
    train_spec = model(train_inputs, training=True)
    test_spec = model(test_inputs, training=False)

    # Train the model
    train_and_test(train_spec, test_spec, n_epoch)

    # Timer end
    now = datetime.now().strftime(time_fmt)
    elapsed_time = datetime.strptime(now, time_fmt) - datetime.strptime(present, time_fmt)
    print('Elapsed time for training: ', elapsed_time)
    print()

    # Final validation by Set5
    test_Set5()