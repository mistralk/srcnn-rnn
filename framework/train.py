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

        Test
        saver.restore(sess, 'tmp/model.ckpt')

        sess.run(test_spec['iterator_init_op'])
        test_loss, test_accuracy = sess.run([test_spec['loss_op'], test_spec['psnr_op']])
        print('------------------------------------')
        print('Test set loss: {}'.format(test_loss))
        print('Test set PSNR: {}'.format(test_accuracy))
        print('------------------------------------')

    file_writer.close()