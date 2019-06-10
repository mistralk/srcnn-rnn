import numpy as np
import tensorflow as tf
import imageio

tf.logging.set_verbosity(tf.logging.ERROR)


def model(inputs, training=False):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        lowres = inputs['lowres']
        ground_truth = inputs['ground_truth']

        # Build a 3-layered network for SR task(2x) with 3x3 conv & ReLU.
        conv1 = tf.layers.conv2d(inputs=lowres, 
                                filters=64, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=tf.nn.relu, 
                                name='conv1')
        
        conv2 = tf.layers.conv2d(inputs=conv1, 
                                filters=64, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=tf.nn.relu, 
                                name='conv2')

        conv3 = tf.layers.conv2d(inputs=conv2, 
                                filters=1, 
                                kernel_size=[3,3], 
                                padding='SAME', 
                                activation=None, 
                                name='conv3')
        
        output = conv3
        output = tf.clip_by_value(output, 0.0, 1.0)

        loss = tf.losses.mean_squared_error(labels=ground_truth, predictions=output)
        psnr = tf.reduce_mean(tf.image.psnr(output, ground_truth, 1.0))

        if training == True:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            training_op = optimizer.minimize(loss)

            loss_summary = tf.summary.scalar('loss', loss)
            psnr_summary = tf.summary.scalar('PSNR', psnr)
            
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