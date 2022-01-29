"""
Core Module for Occlusion Sensitivity
"""
import math

import numpy as np
import cv2

from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.image import apply_grey_patch
from tf_explain.utils.saver import save_rgb


class OcclusionSensitivity:

    """
    Perform Occlusion Sensitivity for a given input
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(
        self,
        validation_data,
        interpreter,
        class_index,
        patch_size=60,
        colormap=cv2.COLORMAP_VIRIDIS,
    ):
        """
        Compute Occlusion Sensitivity maps for a specific class index.
        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            interpreter (tf_lite model): tf.keras model to inspect
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
            colormap (int): OpenCV Colormap to use for heatmap visualization
        Returns:
            np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
        """
        images, _ = validation_data
        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(interpreter, image, class_index, patch_size)
                for image in images
            ]
        )

        heatmaps = np.array(
            [
                heatmap_display(heatmap, image, colormap)
                for heatmap, image in zip(sensitivity_maps, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    def get_sensitivity_map(self, interpreter, image, class_index, patch_size):
        """
        Compute sensitivity map on a given image for a specific class index.
        Args:
            interpreter (tf_lite Model): tf.keras model to inspect
            image:
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
        Returns:
            np.ndarray: Sensitivity map with shape (H, W, 3)
        """
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        patches = np.array([
            apply_grey_patch(image, top_left_x, top_left_y, patch_size)
            for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
        ])

        coordinates = [
            (index_y, index_x)
            for index_x in range(
                sensitivity_map.shape[1]  # pylint: disable=unsubscriptable-object
            )
            for index_y in range(
                sensitivity_map.shape[0]  # pylint: disable=unsubscriptable-object
            )
        ]
        print(interpreter.get_input_details())
        # predictions = model.predict(np.array(patches), batch_size=self.batch_size)
        # 单次处理
        predictions = []
        # for patch in patches:
        #     interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [patch])
        #     interpreter.invoke()
        #     prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        #     predictions.append(prediction)
        
        # 这里将单次处理改为批处理，否则处理速度太慢
        # todo 依然较慢，要考虑如何加快推理速度
        # interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'],[len(patches), 300, 300, 3])
        interpreter.allocate_tensors()
        for patch in patches:
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [patch])
            interpreter.invoke()
            preds = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            predictions.extend(preds)

        target_class_predictions = [
            prediction[class_index] for prediction in predictions
        ]
        # 图像越暗，置信度越高，说明越亮的地方颜值越差
        for (index_y, index_x), confidence in zip(
            coordinates, target_class_predictions
        ):
            sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.
        Args:
            grid (numpy.ndarray): Grid of all heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)
