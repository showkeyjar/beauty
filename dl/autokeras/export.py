import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops

from tensorflow.keras.models import load_model
import autokeras as ak
"""
将keras模型转换为tensorflow lite
"""
directory = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"

df_rates = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv", header=None, names=['filename', 'score'])
df_rates = df_rates[df_rates['filename'].str.find("AF")>=0]
df_rates['score'] = df_rates['score'].astype(int)
df_rates_mean = df_rates.groupby('filename').mean()
df_rates_mean.reset_index(inplace=True)


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def representative_dataset():
    global df_rates_mean
    # images = df_rates_mean.apply(load_image, axis=1)
    # imgs = images.to_numpy(dtype=np.float32)
    # imgs = np.asarray(images.values).astype(np.float32)
    df_samples = df_rates_mean.loc[:16]
    image_paths = [os.path.join(directory, fname) for fname in df_samples["filename"].values]
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(lambda x: path_to_image(x, (350, 350), 3, "bilinear"))

    # for data in rp_dataset.batch(1).take(16):
    for data in img_ds.batch(1).take(16):
        yield [tf.dtypes.cast(data, tf.float32)]
        # yield [tf.dtypes.cast(data, tf.float64)]
        # yield [data]


model = load_model("model_beauty_v1", custom_objects=ak.CUSTOM_OBJECTS)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# quantized = False
quantized = True

if quantized:
    # 整数级量化(其实目前输出的模型已经很小了，没有必要进一步量化)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.float16]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    model_name = "model_beauty_q_v1"
else:
    model_name = "model_beauty_v1"

tflite_model = converter.convert()

# Save the model.
with open(model_name + '.tflite', 'wb') as f:
  f.write(tflite_model)
