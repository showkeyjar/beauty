import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import autokeras as ak
from tensorflow_model_optimization.sparsity import keras as sparsity

"""
pip install --user --upgrade tensorflow-model-optimization

参考：https://medium.com/@chengweizhang2012/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization-adcfa9c9fe94
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

# Backend agnostic way to save/restore models
# _, keras_file = tempfile.mkstemp('.h5')
# print('Saving model to: ', keras_file)
# tf.keras.models.save_model(model, keras_file, include_optimizer=False)

# Load the serialized model
# loaded_model = tf.keras.models.load_model(keras_file)
loaded_model = load_model("model_beauty_v1", custom_objects=ak.CUSTOM_OBJECTS)

num_train_samples = len(df_rates_mean)
batch_size = 16

epochs = 4
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss='mse',
    optimizer='sgd',
    metrics=[tf.keras.metrics.MeanSquaredError()])


# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir="logs/", profile_batch=0)
]

image_paths = [os.path.join(directory, fname) for fname in df_rates_mean["filename"].values]
path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
img_ds = path_ds.map(lambda x: path_to_image(x, (350, 350), 3, "bilinear"))

x_train, x_test, y_train, y_test = train_test_split(img_ds, df_rates_mean['score'].values, test_size=0.25, random_state=42)

new_pruned_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test mse:', score[1])

final_model = sparsity.strip_pruning(new_pruned_model)
final_model.summary()

tf.keras.models.save_model(final_model, "model_beauty_pruned_v1", include_optimizer=False)

