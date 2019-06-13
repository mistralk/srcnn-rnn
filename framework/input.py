import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def preprocess_image(image):
    image = tf.read_file(image)
    image = tf.image.decode_image(image, channels=3)

    grayscaled = tf.image.rgb_to_grayscale(image)

    ground_truth = tf.image.random_crop(grayscaled, [32, 32, 1])
    ground_truth = tf.cast(ground_truth, tf.float32)
    ground_truth /= 255.0

    #lowres = tf.image.resize_images(ground_truth, [16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    downsampled = tf.image.resize_images(ground_truth, [16, 16], method=tf.image.ResizeMethod.BICUBIC)
    lowres = tf.image.resize_images(downsampled, [32, 32], method=tf.image.ResizeMethod.BICUBIC)
    
    ground_truth = tf.clip_by_value(ground_truth, 0.0, 1.0)
    lowres = tf.clip_by_value(lowres, 0.0, 1.0)

    return (lowres, ground_truth)


def input_dataset(dataset_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)
    dataset_length = len(dataset_list)

    dataset = dataset.map(preprocess_image, num_parallel_calls=12)
    dataset = dataset.shuffle(buffer_size=dataset_length)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_initializable_iterator()

    (lowres, ground_truth) = iterator.get_next()
    init_op = iterator.initializer

    inputs = {
        'lowres': lowres,
        'ground_truth': ground_truth,
        'dataset_length': dataset_length,
        'iterator_init_op': init_op
    }

    return inputs