from typing import Iterator, List, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
# from tensorflow.keras.applications import EfficientNetB0

import tensorflow as tf

"""
参考：https://rosenfelder.ai/keras-regression-efficient-net/
# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
pip install tensorflow-addons

下载 efficientnetv2 预训练模型：
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-s.tgz
wget https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-s-imagenet.h5
"""

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

directory = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"

df_rates = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv", header=None, names=['filename', 'score'])
df_rates = df_rates[df_rates['filename'].str.find("AF")>=0]
df_rates['score'] = df_rates['score'].astype(int)
df_rates.replace([np.inf, -np.inf], np.nan, inplace=True)
df_rates['score'].fillna(0, inplace=True)
df_rates_mean = df_rates.groupby('filename').mean()
df_rates_mean.reset_index(inplace=True)
df_rates_mean['level'] = df_rates['score'].round().astype(int)
df_rates_mean['level'].fillna(0, inplace=True)

# use the downloaded and converted newest EfficientNet wheights
# 这里暂未找到合适的预训练模型，暂不做迁移学习
# model = EfficientNetB0(include_top=False, input_tensor=inputs)
pre_model = keras.models.load_model('pretrain_models/efficientnetv2-s-imagenet.h5')
# Freeze the pretrained weights
pre_model.trainable = False
input_shape = (300, 300, 3)
input_size = (300, 300)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Accepts a Pandas DataFrame and splits it into training, testing and validation data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    # df.reset_index(inplace=True)
    train, val = train_test_split(df, test_size=0.3, random_state=1, stratify=df['level'])  # split the data with a validation size o 30%
    # train.reset_index(inplace=True)
    train, test = train_test_split(train, test_size=0.3, random_state=1, stratify=train['level'])  # split the data with an overall  test size of 10%
    train.drop('level', axis=1, inplace=True)
    val.drop('level', axis=1, inplace=True)
    test.drop('level', axis=1, inplace=True)
    print("shape train: ", train.shape)  # type: ignore
    print("shape val: ", val.shape)  # type: ignore
    print("shape test: ", test.shape)  # type: ignore

    print("Descriptive statistics of train:")
    print(train.describe())  # type: ignore
    return train, val, test  # type: ignore


def create_generators(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[Iterator, Iterator, Iterator]:
    """Accepts four Pandas DataFrames: all your data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerators. Within this function you can also visualize the augmentations of the ImageDataGenerators.

    Parameters
    ----------
    df : pd.DataFrame
        Your Pandas DataFrame containing all your data.
    train : pd.DataFrame
        Your Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Your Pandas DataFrame containing your validation data.
    test : pd.DataFrame
        Your Pandas DataFrame containing your testing data.

    Returns
    -------
    Tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    global input_size
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        validation_split=0.3,
    )  # create an ImageDataGenerator with multiple image augmentations
    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )  # except for rescaling, no augmentations are needed for validation and testing generators
    test_generator = ImageDataGenerator(rescale=1.0 / 255)

    # batch_size=32 会导致 oom
    train_generator = train_generator.flow_from_dataframe(
        dataframe=train,
        x_col="image_location",  # this is where your image data is stored
        y_col="score",  # this is your target feature
        class_mode="raw",  # use "raw" for regressions
        target_size=input_size,
        batch_size=16,  # increase or decrease to fit your GPU
    )

    validation_generator = validation_generator.flow_from_dataframe(
        dataframe=val, x_col="image_location", y_col="score", class_mode="raw", target_size=input_size, batch_size=16,
    )
    test_generator = test_generator.flow_from_dataframe(
        dataframe=test, x_col="image_location", y_col="score", class_mode="raw", target_size=input_size, batch_size=16,
    )
    return train_generator, validation_generator, test_generator


def get_callbacks(model_name: str) -> List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]:
    """Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
        "logs/scalars/" + model_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir)
    # use tensorboard --logdir logs/scalars in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_score_loss",
        min_delta=0.5,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    model_checkpoint_callback = ModelCheckpoint(
        "./data/models/" + model_name + ".h5",
        monitor="val_score_loss",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )  # saving eff_net takes quite a bit of time
    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]


def score_loss(y_true, y_pred):
    """
    由于数据集分布不平衡，自定义loss
    """
    abs_score = tf.abs(y_true - y_pred)
    # 超过 float32 大数导致 nan
    # sq_score = tf.math.square(abs_score)
    # 由于极值总是不准确，加大极值的惩罚力度
    mult_score = tf.math.multiply(tf.abs(y_true - 3), 2) + 1
    new_score = tf.math.multiply(abs_score, mult_score)
    # 分值越高越重视(惩罚力度越大)
    beauty_loss = tf.reduce_mean(new_score)
    return beauty_loss


def run_model(
    model_name: str,
    lr: float,
    train_generator: Iterator,
    validation_generator: Iterator,
    test_generator: Iterator,
) -> Model:
    """This function runs a keras model with the Ranger optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and afterwards one final time on the test generator.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.
    lr : float
        Learning rate.
    train_generator : Iterator
        keras ImageDataGenerators for the training data.
    validation_generator : Iterator
        keras ImageDataGenerators for the validation data.
    test_generator : Iterator
        keras ImageDataGenerators for the test data.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history. For an example
        see plot_results().
    """
    global strategy, pre_model, input_shape
    callbacks = get_callbacks(model_name)
    radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    optimizer = ranger

    # inputs = layers.Input(
    #     shape=input_shape
    # )
    # input shapes of the images with efficientnetv2-s

    # Rebuild top
    # x = layers.GlobalAveragePooling1D(name="avg_pool")(pre_model.output)
    # x = layers.BatchNormalization()(x)
    # 因为数据集较小，将dropout比率控制在0.2
    # top_dropout_rate = 0.2
    # x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    # outputs = layers.Dense(1, name="pred")(pre_model.output)
    # 由于EfficientNetV2 输出已经是(None, 1000),所以这里不再增加处理层
    # outputs = pre_model.output

    # with strategy.scope():
    inputs = pre_model.input
    x = layers.BatchNormalization()(pre_model.output)
    outputs = layers.Dense(1, name="pred")(x)
    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNetV2")
    # model = pre_model
    # model.trainable = True
    # model.summary()
    model.compile(
        optimizer=optimizer, loss=score_loss, metrics=[score_loss]
    )
    # 可暂不使用分布式数据集
    # train_dataset = tf.data.Dataset.from_generator(lambda:train_generator, output_types=(tf.float32, tf.float32), output_shapes=([None, 224, 224, 3], [None, 1]))
    # validation_dataset = tf.data.Dataset.from_generator(lambda:validation_generator, output_types=(tf.float32, tf.float32), output_shapes=([None, 224, 224, 3], [None, 1]))
    # test_dataset = tf.data.Dataset.from_generator(lambda:test_generator, output_types=(tf.float32, tf.float32), output_shapes=([None, 224, 224, 3], [None, 1]))
    # train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # validation_dataset = strategy.experimental_distribute_dataset(validation_dataset)
    # test_dataset = strategy.experimental_distribute_dataset(test_dataset)
    # fit_generator 已过时，fit支持 generator
    model.fit(
        train_generator,
        epochs=1000,
        validation_data=validation_generator,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=16,  # adjust this according to the number of CPU cores of your machine
    )

    model.evaluate(
        test_generator, callbacks=callbacks,
    )
    return model  # type: ignore


def run(small_sample=False):
    """Run all the code of this file.

    Parameters
    ----------
    small_sample : bool, optional
        If you just want to check if the code is working, set small_sample to True, by default False
    """
    global df_rates_mean
    df = df_rates_mean.copy()
    df["image_location"] = df["filename"].apply(lambda x: directory + x) # add the correct path for the image locations.
    if small_sample == True:
        df = df.iloc[0:1000]  # set small_sampe to True if you want to check if your code works without long waiting
    train, val, test = split_data(df)  # split your data
    
    train_generator, validation_generator, test_generator = create_generators(
        df=df, train=train, val=val, test=test
    )

    # 由于太容易过拟合，减小学习率（但学习率过小会导致loss计算结果nan）
    eff_net_model = run_model(
        model_name="eff_net",
        lr=0.1,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )
    eff_net_model.save("ef_v2_model.h5")
    # plot_results(small_cnn_history, eff_net_history, mean_baseline)


if __name__ == "__main__":
    # ps -ef|grep "python train_reg" | awk '{print $2}' | xargs kill -9
    run(small_sample=False)
