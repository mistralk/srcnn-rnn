import numpy as np
import tensorflow as tf
import os
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)


def train_and_test(train_spec, test_spec, n_epoch):
    # Set logging directory with timestamps
    # logging code is from "Hands on ML" book
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    iterator_init_op = train_spec['iterator_init_op']
    train_op = train_spec['train_op']
    loss_op = train_spec['loss_op']
    psnr_op = train_spec['psnr_op']
    summary_op = train_spec['summary_op']

    with tf.Session() as sess:
        sess.run(init)
        
        best_accuracy = -99999.0

        
        for item in tf.trainable_variables():
            print(item)

        # Training
        for epoch in range(n_epoch):
            sess.run(train_spec['iterator_init_op'])

            if epoch % 10 == 0:
                _, loss, accuracy, summary = sess.run([train_op, loss_op, psnr_op, summary_op])
                file_writer.add_summary(summary, global_step=epoch)
    
            else:
                _, loss, accuracy = sess.run([train_op, loss_op, psnr_op])
            
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                save_path = saver.save(sess, 'tmp/model.ckpt')

            print('epoch #{} PSNR:{} LOSS:{}'.format(epoch, accuracy, loss))

        # Test
        saver.restore(sess, 'tmp/model.ckpt')
        
         # Get parameter from tf variables
        default_graph = tf.get_default_graph()
        conv1_kernel = default_graph.get_tensor_by_name('model/conv2d/kernel:0').eval()
        conv1_bias = default_graph.get_tensor_by_name('model/conv2d/bias:0').eval()
        conv2_kernel = default_graph.get_tensor_by_name('model/conv2d_1/kernel:0').eval()
        conv2_bias = default_graph.get_tensor_by_name('model/conv2d_1/bias:0').eval()
        conv3_kernel = default_graph.get_tensor_by_name('model/conv2d_2/kernel:0').eval()
        conv3_bias = default_graph.get_tensor_by_name('model/conv2d_2/bias:0').eval()
        conv4_kernel = default_graph.get_tensor_by_name('model/conv2d_3/kernel:0').eval()
        conv4_bias = default_graph.get_tensor_by_name('model/conv2d_3/bias:0').eval()
        conv5_kernel = default_graph.get_tensor_by_name('model/conv2d_4/kernel:0').eval()
        conv5_bias = default_graph.get_tensor_by_name('model/conv2d_4/bias:0').eval()

        # Save parameters as numpy array
        np.save("conv1_kernel", conv1_kernel)
        np.save("conv1_bias", conv1_bias)
        np.save("conv2_kernel", conv2_kernel)
        np.save("conv2_bias", conv2_bias)
        np.save("conv3_kernel", conv3_kernel)
        np.save("conv3_bias", conv3_bias)
        np.save("conv4_kernel", conv3_kernel)
        np.save("conv4_bias", conv3_bias)
        np.save("conv5_kernel", conv3_kernel)
        np.save("conv5_bias", conv3_bias)
        
        sess.run(test_spec['iterator_init_op'])
        test_loss, test_accuracy = sess.run([test_spec['loss_op'], test_spec['psnr_op']])
        print('------------------------------------')
        print('Test set loss: {}'.format(test_loss))
        print('Test set PSNR: {}'.format(test_accuracy))
        print('------------------------------------')

    file_writer.close()