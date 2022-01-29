import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import autokeras as ak

"""
keras模型部署到Android
https://www.codenong.com/cs105821253/
keras多gpu支持
https://keras.io/guides/distributed_training/

不能使用 ImageDataGenerator，否则训练时会卡住
"""


face_datagen = ImageDataGenerator(
    horizontal_flip=True,
    validation_split=0.3
)

face_path = "/opt/data/SCUT-FBP5500_v2/score_data/fm"

train_generator = face_datagen.flow_from_directory(
        face_path,
        target_size=(350, 350),
        batch_size=32,
        class_mode="sparse",
        subset='training')

val_generator = face_datagen.flow_from_directory(
        face_path,
        target_size=(350, 350),
        batch_size=32,
        class_mode="sparse",
        subset='validation')


def callable_iterator(generator):
    for img_batch, targets_batch in generator:
        yield img_batch, targets_batch


# train_dataset = tf.data.Dataset.from_generator(lambda: callable_iterator(train_generator),
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([None, 350, 350, 3], [None,])
# )

# val_dataset = tf.data.Dataset.from_generator(lambda: callable_iterator(val_generator),
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([None, 350, 350, 3], [None,])
# )


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    face_path,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(350, 350),
    shuffle=True,
    seed=96,
    validation_split=0.3,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    face_path,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=16,
    image_size=(350, 350),
    shuffle=True,
    seed=96,
    validation_split=0.3,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
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

# reg = ak.ImageRegressor(overwrite=True, max_trials=1)
# reg = ak.ImageRegressor(overwrite=True)
# 不要覆盖，继续训练
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

reg = ak.ImageRegressor(max_trials=50, distribution_strategy=strategy)


if __name__=="__main__":
    # ps -ef|grep "python train.py" | awk '{print $2}' | xargs kill -9
    # reg.fit(train_dataset, epochs=100)
    reg.fit(train_dataset, epochs=200)
    #Evaluate the best model.
    print(reg.evaluate(val_dataset))

    model = reg.export_model()

    try:
        model.save("model_autokeras", save_format="tf")
    except Exception:
        model.save("model_autokeras.h5")
