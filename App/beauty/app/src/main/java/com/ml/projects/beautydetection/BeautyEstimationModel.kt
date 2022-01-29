package com.ml.projects.beautydetection

// import android.util.Log
import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Helper class for age estimation model
class BeautyEstimationModel {

    // Input image size for our model
    private val inputImageSize = 300

    // Image processor for model inputs.
    // Resize + Normalize
    // 这里resize后，理论上应该是 1 * 300 * 300 * 3 = 367500, 但实际却是 1470000, 刚好是预期输入的4倍
    // 是由于float为4字节，但int为1字节
    // tensorflow lite 对模型进行转换后，按正常情况应该自动适应 kotlin bitmap 的输入格式
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f))
                    .build()

    // The model returns a normalized value for the age i.e in range ( 0 , 1 ].
    // To get the age, we multiply the model's output with p.
    // private val p = 116

    // Time taken by the model ( in milliseconds ) to perform the inference.
    var inferenceTime : Long = 0

    // Interpreter object to use the TFLite model.
    var interpreter : Interpreter? = null

    // Given an input image, return the estimated score.
    // Note: This is a suspended function, and will run within a CoroutineScope.
    suspend fun predictScore(image: Bitmap) = withContext( Dispatchers.Main ) {
        val start = System.currentTimeMillis()
        // Input image tensor shape -> [ 1 , 300 , 300 , 3 ]
        val tensorInputImage = TensorImage.fromBitmap(image)
        // Output tensor shape -> [ 1 , 1 ]
        val scoreOutputArray = Array(1){ FloatArray(1) }
        val processedImageBuffer = inputImageProcessor.process(tensorInputImage).buffer
        // todo 整数量化后的模型要注意做转换
        interpreter?.run(
                processedImageBuffer,
                scoreOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        // Log.i("Score", scoreOutputArray[0].toString())
        return@withContext scoreOutputArray[0][0]
    }
}



