import tool_util
from data_util import GeneratorEnqueuer
import numpy as np
import os, glob, cv2
import tensorflow as tf
import time
import random

logger = tool_util.logger

tf.app.flags.DEFINE_string('training_image_path', './kaggle/train',
                           'training images to use')
tf.app.flags.DEFINE_string('training_label_file', './VOCdevkit/VOC2007/ImageSets/labels.txt',
                           'training labels to use')
FLAGS = tf.app.flags.FLAGS

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def get_images(file_path=FLAGS.training_image_path):
    filelist = []
    for ext in ['jpg', 'JPG', 'jpeg', 'png']:
        filelist.extend(glob.glob(os.path.join(file_path, '*.{}'.format(ext))))
    return filelist

def get_lables():
    labellist = {}
    with open(FLAGS.training_label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key,value = line.split(' ')
            labellist[key] = int(value)

def onehot(index):
    """ It creates a one-hot vector with a 1.0 in
            position represented by index
    """
    onehot = np.zeros(3)
    onehot[index] = 1.0
    return onehot

def random_rgb2gray(img, p=0.08):

    if random.random() < p:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def random_horizontal_flip(img, p=0.5):

    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]

    return img

def random_exposure(img, delta=20, p=0.5):
    if random.random() < p:
        exposure = np.random.uniform(-delta, delta)
        img_type = img.dtype
        img = img.astype(np.float32)
        img += exposure
        img = np.clip(img, 0, 255)
        img = img.astype(img_type)
    return img

#todo:默认是0-255图像
def random_hsv(img, hue = 2, saturation=5, value=0, p=0.5):
    if random.random() < p:
        h = np.random.uniform(-hue, hue)
        s = np.random.uniform(-saturation, saturation)
        # v = np.random.uniform(-value, value)
        img_hsv= cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_type = img.dtype
        img_hsv = img_hsv.astype(np.float32)
        #default img 0-255, hue 0-180
        img_hsv[..., 0] = np.clip(img_hsv[..., 0]+h, 0, 180)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1]+s, 0, 255)
        # img_hsv[..., 2] = np.clip(img_hsv[..., 2]+v, 0, 255)
        img_hsv = img_hsv.astype(img_type)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img

def generator(input_size=224, batch_size=32):
    imagelist = np.array(get_images())
    index = np.arange(0, imagelist.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        lables = []
        for i in index:
            try:
               image = cv2.imread(imagelist[i])
               if image is None:
                   continue
               image = random_horizontal_flip(image)
               image = random_hsv(image)
               image = random_exposure(image)
               image = random_rgb2gray(image)
               h,w,_=image.shape
               assert h>0 and w>0
               image = cv2.resize(image, (input_size, input_size))

               #0 -- cat 1--dog
               if os.path.basename(imagelist[i]).split('.')[0]=='cat':
                   lables.append(onehot(0))
               elif os.path.basename(imagelist[i]).split('.')[0]=='dog':
                   lables.append(onehot(1))
               else:
                   logger.error(imagelist[i]+' wrong name')
                   continue
               images.append(image[:, :, ::-1].astype(np.float32))
               if len(images) == batch_size:
                   yield images, lables
                   images = []
                   lables = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__=='__main__':
    generator()
