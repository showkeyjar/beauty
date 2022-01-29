import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from keras.preprocessing.image import ImageDataGenerator


def mnist_x(x_orig, mdl_input_dims, is_training):

    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / x_orig.dtype.max

    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]
    # print("height_width", height_width)
    # training transformations
    # 这里为何选择20的尺寸？考虑到新图像比较大，改为32
    crop_size = 32
    central_fraction = np.mean(crop_size / np.array(x_orig.shape.as_list()[1:-1]))
    # print("central_fraction", central_fraction)
    if is_training:
        crop_heigh_width = [int(height_width[0] * central_fraction), int(height_width[1] * central_fraction)]
        x1 = tf.image.central_crop(x_orig, central_fraction)
        x1 = tf.image.resize(x1, crop_heigh_width)
        x2 = tf.image.random_crop(x_orig, tf.concat((tf.shape(x_orig)[:1], crop_heigh_width, [n_chans]), axis=0))
        x = tf.stack([x1, x2])
        x = tf.transpose(x, [1, 0, 2, 3, 4])
        i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(x)[0]))
        x = tf.map_fn(lambda y: y[0][y[1]], (x, i), dtype=tf.float32)
        x = tf.image.resize(x, height_width)
    # testing transformations
    else:
        x = tf.image.central_crop(x_orig, central_fraction)
        x = tf.image.resize(x, height_width)
    return x


def mnist_gx(x_orig, mdl_input_dims, is_training, sample_repeats):

    # if not training, return a constant value--it will unused but needs to be same shape to avoid TensorFlow errors
    if not is_training:
        return tf.zeros([0] + mdl_input_dims)

    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / x_orig.dtype.max

    # repeat samples accordingly
    x_orig = tf.tile(x_orig, [sample_repeats] + [1] * len(x_orig.shape.as_list()[1:]))

    # get common shapes
    height_width = mdl_input_dims[:-1]
    n_chans = mdl_input_dims[-1]

    # random rotation
    rad = 2 * np.pi * 25 / 360
    x_rot = tf.contrib.image.rotate(x_orig, tf.random.uniform(shape=tf.shape(x_orig)[:1], minval=-rad, maxval=rad))
    gx = tf.stack([x_orig, x_rot])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

    # random crops
    x1 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [16, 16], [n_chans]), axis=0))
    x2 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [20, 20], [n_chans]), axis=0))
    x3 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [24, 24], [n_chans]), axis=0))
    gx = tf.stack([tf.image.resize(x1, height_width),
                   tf.image.resize(x2, height_width),
                   tf.image.resize(x3, height_width)])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

    # apply random adjustments
    def rand_adjust(img):
        img = tf.image.random_brightness(img, 0.4)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        if img.shape.as_list()[-1] == 3:
            img = tf.image.random_saturation(img, 0.6, 1.4)
            img = tf.image.random_hue(img, 0.125)
        return img

    gx = tf.map_fn(lambda y: rand_adjust(y), gx, dtype=tf.float32)
    return gx


def pre_process_data(ds, is_training, **kwargs):
    """
    :param ds: TensorFlow Dataset object
    :param is_training: indicator to pre-processing function
    :return: the passed in data set with map pre-processing applied
    """
    # apply pre-processing function for given data set and run-time conditions
    
    return ds.map(
        lambda d_image: {
            'x': mnist_x(d_image, mdl_input_dims=kwargs['mdl_input_dims'], is_training=is_training),
            'gx': mnist_gx(d_image, mdl_input_dims=kwargs['mdl_input_dims'], is_training=is_training, sample_repeats=kwargs['num_repeats']),
            'label': 1
        },
        num_parallel_calls=tf.data.experimental.AUTOTUNE
        )


def configure_data_set(ds, batch_size, is_training, **kwargs):
    """
    :param ds: TensorFlow data set object
    :param batch_size: batch size
    :param is_training: indicator to pre-processing function
    :return: a configured TensorFlow data set object
    """
    # enable shuffling and repeats
    # ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(1)

    # batch the data before pre-processing
    # ds = ds.batch(batch_size)

    # pre-process the data set
    with tf.device('/cpu:0'):
        ds = pre_process_data(ds, is_training, **kwargs)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load(**kwargs):
    """
    todo 检查 train_generator 的 shape
    :param data_set_name: data set name--call tfds.list_builders() for options
    :return:
        train_ds: TensorFlow Dataset object for the training data
        test_ds: TensorFlow Dataset object for the testing data
        info: data set info object
    """
    data_path = "/opt/data/SCUT-FBP5500_v2/skin/"
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_path, labels=None, color_mode='rgb', image_size=(80, 96), 
    #     shuffle=True, seed=96, validation_split=0.3, subset='training', interpolation='bilinear', **kwargs
    # )

    # test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_path, labels=None, color_mode='rgb', image_size=(80, 96), 
    #     shuffle=True, seed=96, validation_split=0.3, subset='validation', interpolation='bilinear', **kwargs
    # )
    img_h = 96
    img_w = 80
    batch_size = kwargs['batch_size']

    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3)

    train_generator = datagen.flow_from_directory(
            data_path,
            target_size=(img_w, img_h),
            batch_size=batch_size, 
            classes=['test'],
            class_mode=None,
            seed=96,
            subset='training')

    test_generator = datagen.flow_from_directory(
            data_path,
            target_size=(img_w, img_h),
            batch_size=batch_size, 
            classes=['test'],
            class_mode=None,
            seed=96,
            subset='validation')
    
    out_type = tf.uint8
    out_shape = tf.TensorShape([None, img_w, img_h, 3])

    train_set = tf.data.Dataset.from_generator(lambda: train_generator, out_type, out_shape)
    test_set = tf.data.Dataset.from_generator(lambda: test_generator, out_type, out_shape)

    train_ds = configure_data_set(ds=train_set, is_training=True, **kwargs)
    test_ds = configure_data_set(ds=test_set, is_training=False, **kwargs)

    return train_ds, test_ds
