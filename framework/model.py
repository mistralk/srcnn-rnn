import numpy as np
import tensorflow as tf
import imageio

tf.logging.set_verbosity(tf.logging.ERROR)


def cnn_layers(x_t, h_prev):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # input x_t: 32x32x2
        # conv patch: 3x3x2x32
        # output z_t: 32x32x32
        # shared variable: 
        z_t = tf.layers.conv2d(inputs=x_t,
                            filters=32,
                            kernel_size=[3,3],
                            padding='SAME',
                            activation=tf.nn.relu,)
        
        # input z_t: 32x32x32
        # conv patch: 3x3x32x32
        # output conv_z: 32x32x32
        # shared variable: w_z
        conv_z = tf.layers.conv2d(inputs=z_t,
                                filters=32,
                                kernel_size=[3,3],
                                padding='SAME',
                                activation=None)

        # input h_(t-1): 32x32x32
        # conv patch: 3x3x32x32
        # output conv_h: 32x32x32
        # shared variable: w_h
        conv_h = tf.layers.conv2d(inputs=h_prev,
                                filters=32,
                                kernel_size=[3,3],
                                padding='SAME',
                                activation=None)
        
        h_t = tf.nn.relu(conv_h + conv_z)

        # input h_t: 32x32x32
        # conv patch: 3x3x32x32
        # output hh: 32x32x32
        # shared variable: w_hh
        hh = tf.layers.conv2d(inputs=h_t,
                                filters=32,
                                kernel_size=[3,3],
                                padding='SAME',
                                activation=tf.nn.relu)

        # input hh: 32x32x32
        # conv patch: 3x3x32x1
        # output y_t: 32x32x1
        # shared variable: 
        y_t = tf.layers.conv2d(inputs=hh,
                            filters=1,
                            kernel_size=[3,3],
                            padding='SAME',
                            activation=tf.nn.relu)

        return y_t, h_t


def model(inputs, training=False):
    lowres = inputs['lowres']
    ground_truth = inputs['ground_truth']

    #lowres = tf.image.resize_images(lowres, [32, 32], method=tf.image.ResizeMethod.BICUBIC)

    y0 = lowres
    h0 = tf.zeros([tf.shape(lowres)[0], 32, 32, 32])

    y1, h1 = cnn_layers(x_t=tf.concat([lowres, y0], 3), h_prev=h0) # t=1
    y2, h2 = cnn_layers(x_t=tf.concat([lowres, y1], 3), h_prev=h1) # t=2
    y3, h3 = cnn_layers(x_t=tf.concat([lowres, y2], 3), h_prev=h2) # t=3

    output = y3
    output = tf.clip_by_value(output, 0.0, 1.0)

    loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
    psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

    if training == True:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        training_op = optimizer.minimize(loss)

        with tf.name_scope('accuracy'):
            loss_summary = tf.summary.scalar('loss', loss)
            psnr_summary = tf.summary.scalar('PSNR', psnr)
        
        with tf.name_scope('preview'):
            img_gt_summary = tf.summary.image("ground truth", ground_truth, max_outputs=1)
            img_output_summary = tf.summary.image("SR result", output, max_outputs=1)
            img_input_summary = tf.summary.image("lowres input", lowres, max_outputs=1)

    # Save the model specification
    spec = inputs
    spec['output'] = output
    spec['loss_op'] = loss
    spec['psnr_op'] = psnr

    if training == True:
        spec['train_op'] = training_op
        spec['summary_op'] = tf.summary.merge_all()

    return spec


def reuse_with_metric(lowres_img, groundtruth_img, ckpt_path):

    w = lowres_img.shape[1]
    h = lowres_img.shape[0]

    lowres_img = np.asarray([lowres_img])
    lowres_img = np.reshape(lowres_img, [1, h, w, 1])
    groundtruth_img = np.asarray([groundtruth_img])
    groundtruth_img = np.reshape(groundtruth_img, [1, h, w, 1])

    lowres = tf.placeholder(tf.float32, shape=[1, h, w, 1])
    ground_truth = tf.placeholder(tf.float32, shape=[1, h, w, 1])

    y0 = lowres
    h0 = tf.zeros([1, h, w, 32])

    y1, h1 = cnn_layers(x_t=tf.concat([lowres, y0], 3), h_prev=h0) # t=1
    y2, h2 = cnn_layers(x_t=tf.concat([lowres, y1], 3), h_prev=h1) # t=2
    y3, h3 = cnn_layers(x_t=tf.concat([lowres, y2], 3), h_prev=h2) # t=3
    
    output = y3
    output = tf.clip_by_value(output, 0, 255)

    loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
    psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 255))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        output_var, accuracy_var = sess.run([output, psnr], feed_dict={lowres: lowres_img, ground_truth: groundtruth_img})

        return output_var[0], accuracy_var


def reuse_without_metric(lowres_img, ckpt_path):

    w = lowres_img.shape[1]
    h = lowres_img.shape[0]

    lowres_img = np.asarray([lowres_img])
    lowres_img = np.reshape(lowres_img, [1, h, w, 1])

    lowres = tf.placeholder(tf.float32, shape=[1, h, w, 1])

    y0 = lowres
    h0 = tf.zeros([1, h, w, 32])

    y1, h1 = cnn_layers(x_t=tf.concat([lowres, y0], 3), h_prev=h0) # t=1
    y2, h2 = cnn_layers(x_t=tf.concat([lowres, y1], 3), h_prev=h1) # t=2
    y3, h3 = cnn_layers(x_t=tf.concat([lowres, y2], 3), h_prev=h2) # t=3
    
    output = y3
    output = tf.clip_by_value(output, 0, 255)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)
        output_var = sess.run(output, feed_dict={lowres: lowres_img})

        return output_var[0]

