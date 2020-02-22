#-*- coding:utf-8 -*-
from alexnet import alexnet
from nets.resnet import resnet_v1
import tensorflow as tf
from tensorflow.contrib import slim
from tool_util import logger
import cv2
import time
import numpy as np
from nets.mobilenet import mobilenet_v2

#only support 224 in this alexnet
tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_string('test_image_path', './kaggle/test','test images to use')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './classify_tmp2/', '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')

import readdata

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name = 'input_images')
        keep_prob = tf.placeholder(tf.float32)
        vn = [v.name for v in tf.trainable_variables()]
        for name in vn:
            print(name)

        input_images = readdata.mean_image_subtraction(input_images)
        with slim.arg_scope(mobilenet_v2.training_scope()):
            outputs, end_points = mobilenet_v2.mobilenet(input_images, is_training=False, num_classes=FLAGS.num_classes)
        # with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        #     outputs, _ = alexnet.alexnet_v2(input_images, num_classes=FLAGS.num_classes, is_training=False)
        # with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=1e-5)):
        #     outputs, end_points = resnet_v1.resnet_v1_50(input_images,  is_training=False, scope='resnet_v1_50', num_classes=FLAGS.num_classes)
            probs = tf.squeeze(end_points['Predictions'])

            logger.debug(tf.shape(probs))


            saver = tf.train.Saver()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                logger.info('Restore from {}'.format(FLAGS.checkpoint_path))
                logger.debug("ckpt_state.model_checkpoint_path:" + ckpt_state.model_checkpoint_path)
                saver.restore(sess, ckpt_state.model_checkpoint_path)

                image_list = readdata.get_images(FLAGS.test_image_path)

                for image_name in image_list:
                    im = cv2.imread(image_name)[:,:,::-1]

                    img = im.copy()

                    im_resized=cv2.resize(img, (FLAGS.input_size, FLAGS.input_size))
                    start_time = time.time()
                    logger.info(image_name)
                    prob = sess.run([probs], feed_dict={input_images:[im_resized]})

                    logger.debug(prob)
                    logger.info('detection time:{:.0f}ms result:(other:{:.3f} chat:{:.3f})'.format(
                        (time.time()-start_time)*1000, prob[0][0], prob[0][1]))


if __name__=='__main__':
    tf.app.run()







