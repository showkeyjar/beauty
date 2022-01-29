import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops

import autokeras as ak

"""
keras模型部署到Android
https://www.codenong.com/cs105821253/
keras多gpu支持
https://keras.io/guides/distributed_training/
数据集加载参考
https://stackoverflow.com/questions/41749398/using-keras-imagedatagenerator-in-a-regression-model

不能使用 ImageDataGenerator，否则训练时会卡住
"""

directory = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"

df_rates = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv", header=None, names=['filename', 'score'])
df_rates = df_rates[df_rates['filename'].str.find("AF")>=0]
df_rates['score'] = df_rates['score'].astype(int)
df_rates_mean = df_rates.groupby('filename').mean()
df_rates_mean.reset_index(inplace=True)

"""
由于tensorflow image_dataset_from_directory 不支持图像回归，自己实现一个简单的版本
"""
def index_df(df,
                    shuffle=True,
                    seed=None):
    global directory
    labels = df["score"].values
    file_paths = [os.path.join(directory, fname) for fname in df["filename"].values]

    if shuffle:
        # Shuffle globally to erase macro-structure
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(file_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(labels)
    return file_paths, labels


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
    """Constructs a dataset of images and labels."""
    # TODO(fchollet): consider making num_parallel_calls settable
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(
        lambda x: path_to_image(x, image_size, num_channels, interpolation))
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
    return img_ds


def image_dataset_from_df(df,
                                 labels='inferred',
                                 label_mode='float',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear'):
    if labels != 'inferred':
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                '`labels` argument should be a list/tuple of integer labels, of '
                'the same size as the number of image files in the target '
                'directory. If you wish to infer the labels from the subdirectory '
                'names in the target directory, pass `labels="inferred"`. '
                'If you wish to get a dataset that only contains images '
                '(no labels), pass `label_mode=None`.')
        if class_names:
            raise ValueError('You can only pass `class_names` if the labels are '
                            'inferred from the subdirectory names in the target '
                            'directory (`labels="inferred"`).')
    if label_mode not in {'int', 'float', 'categorical', 'binary', None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "float", "categorical", "binary", '
            'or None. Received: %s' % (label_mode,))
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,))
    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)
    image_paths, labels = index_df(
        df,
        shuffle=shuffle,
        seed=seed)

    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
            'Found the following classes: %s' % (class_names,))

    image_paths, labels = dataset_utils.get_training_or_validation_split(
        image_paths, labels, validation_split, subset)
    
    if class_names is None:
        class_num = 1
    else:
        class_num = len(class_names)

    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=num_channels,
        labels=labels,
        label_mode=label_mode,
        num_classes=class_num,
        interpolation=interpolation)
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset


train_dataset = image_dataset_from_df(
    df_rates_mean,
    labels="inferred",
    label_mode="float",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(350, 350),
    shuffle=True,
    seed=96,
    validation_split=0.3,
    subset="training",
    interpolation="bilinear"
)


val_dataset = image_dataset_from_df(
    df_rates_mean,
    labels="inferred",
    label_mode="float",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(350, 350),
    shuffle=True,
    seed=96,
    validation_split=0.3,
    subset="validation",
    interpolation="bilinear"
)


# Initialize the image regressor.
# input_node = ak.ImageInput()
# output_node = ak.Normalization()(input_node)
# output_node = ak.ImageAugmentation(horizontal_flip=False)(output_node)
# output_node = ak.ResNetBlock(version="v2")(output_node)
# output_node = ak.RegressionHead()(output_node)
# reg = ak.AutoModel(
#     inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
# )

# 不要覆盖，继续训练
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

reg = ak.ImageRegressor(overwrite=True, max_trials=300, distribution_strategy=strategy)
# reg = ak.ImageRegressor(max_trials=200, distribution_strategy=strategy)


if __name__=="__main__":
    # ps -ef|grep "python train.py" | awk '{print $2}' | xargs kill -9
    # reg.fit(train_dataset, epochs=100)
    reg.fit(train_dataset, epochs=200)
    #Evaluate the best model.
    print(reg.evaluate(val_dataset))

    model = reg.export_model()

    try:
        model.save("model_beauty_v1", save_format="tf")
    except Exception:
        model.save("model_beauty_v1.h5")
