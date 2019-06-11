import numpy as np
import tensorflow as tf
import imageio

tf.logging.set_verbosity(tf.logging.ERROR)


def cnn_layers(x_t, h_prev):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        # input x_t: 32x32x2
        # conv patch: 3x3x2x32
        # output z: 32x32x32
        # shared variable: w_z
        z = tf.layers.conv2d(inputs=x_t,
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
        
        '''
        # shared variable: w_z
        conv_z = tf.layers.conv2d(inputs=z,
                                filters=32,
                                kernel_size=[3,3],
                                padding='SAME',
                                activation=None)
        '''
        h_t = tf.nn.relu(conv_h + z)

        # shared variable: w_hh
        y_t = tf.layers.conv2d(inputs=h_t,
                            filters=1,
                            kernel_size=[3,3],
                            padding='SAME',
                            activation=tf.nn.relu)

        return y_t, h_t


def model(inputs, training=False):
    lowres = inputs['lowres']
    ground_truth = inputs['ground_truth']

    y0 = lowres
    h0 = tf.zeros([tf.shape(lowres)[0], 32, 32, 32])

    #h0 = tf.zeros_like(tf.concat([lowres, lowres], 3))

    y1, h1 = cnn_layers(x_t=tf.concat([lowres, y0], 3), h_prev=h0) # t=1
    y2, h2 = cnn_layers(x_t=tf.concat([lowres, y1], 3), h_prev=h1) # t=2
    y3, h3 = cnn_layers(x_t=tf.concat([lowres, y2], 3), h_prev=h2) # t=3

    output = y3
    output = tf.clip_by_value(output, 0.0, 1.0)

    loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
    psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

    if training == True:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
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


def reuse_model(model_path, model_spec):
    saver = tf.train.Saver()

    lw_images = model_spec['lowres']
    gt_images = model_spec['ground_truth']
    output_images = model_spec['output']
    loss_op = model_spec['loss_op']
    psnr_op = model_spec['psnr_op']

    with tf.Session() as sess:
        saver.restore(sess, model_path)
        sess.run(model_spec['iterator_init_op'])
               
        lw, gt, output, loss, accuracy = sess.run([lw_images, gt_images, output_images, loss_op, psnr_op])

        # Save result images
        for i in range(model_spec['dataset_length']):
            imageio.imwrite(str(i) + '_lowres.png', lw[i])
            imageio.imwrite(str(i) + '_gt.png', gt[i])
            imageio.imwrite(str(i) + '_output.png', output[i])

        print('------------------------------------')
        print('Set5 loss: {}'.format(loss))
        print('Set5 PSNR: {}'.format(accuracy))
        print('------------------------------------')