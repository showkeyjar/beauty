import tensorflow as tf
from tensorflow import keras
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


# Load pretrained model or your own
model = keras.models.load_model('ef_v2_model.h5', compile=False)

# Load a sample image (or multiple ones)
img = tf.keras.preprocessing.image.load_img("/opt/data/SCUT-FBP5500_v2/Images/train/face/AF1031.jpg", target_size=(300, 300))
img = tf.keras.preprocessing.image.img_to_array(img)
data = ([img], None)

# Start explainer
explainer = OcclusionSensitivity()
grid = explainer.explain(data, model, class_index=0, patch_size=4)  # 281 is the tabby cat index in ImageNet

explainer.save(grid, ".", "occ_sens.png")
