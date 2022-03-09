package com.ml.projects.beautydetection

// import android.util.Log
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.chaquo.python.Python
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.ByteArrayOutputStream

// Helper class for face skin estimation model
class SkinEstimationModel {

    private val highAccuracyOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .build()
    // Input image size for our model
    private val inputImageWidth = 80
    private val inputImageHeight = 96
    private lateinit var cheekBitMap: Bitmap

    // Image processor for model inputs.
    // Resize + Normalize
    // 这里resize后，理论上应该是 1 * 80 * 96 * 3 = 23040, 但实际却是 1470000, 刚好是预期输入的4倍
    // 是由于float为4字节，但int为1字节
    // tensorflow lite 对模型进行转换后，按正常情况应该自动适应 kotlin bitmap 的输入格式
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add(ResizeOp(inputImageWidth, inputImageHeight, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f))
                    .build()

    // Interpreter object to use the TFLite model.
    var interpreter : Interpreter? = null


    private fun bitmapToString(inputFace: Bitmap): String {
        val stream = ByteArrayOutputStream()
        inputFace.compress(Bitmap.CompressFormat.PNG, 90, stream)
        val pyImage = stream.toByteArray()
        return Base64.encodeToString(pyImage, Base64.DEFAULT)
    }

    private fun cutImage(inputFace: Bitmap, py:Python){
        val image = InputImage.fromBitmap(inputFace, 0)
        val detector = FaceDetection.getClient(highAccuracyOpts)
        val result = detector.process(image)
            .addOnSuccessListener { faces ->
                // Task completed successfully
                for (face in faces) {
                    val bounds = face.boundingBox
                    val rotY = face.headEulerAngleY // Head is rotated to the right rotY degrees
                    val rotZ = face.headEulerAngleZ // Head is tilted sideways rotZ degrees

                    // If landmark detection was enabled (mouth, ears, eyes, cheeks, and
                    // nose available):
                    val leftEar = face.getLandmark(FaceLandmark.LEFT_EAR)
                    leftEar?.let {
                        val leftEarPos = leftEar.position
                    }

                    // If contour detection was enabled:
                    val leftEyeContour = face.getContour(FaceContour.LEFT_EYE)?.points
                    val upperLipBottomContour = face.getContour(FaceContour.UPPER_LIP_BOTTOM)?.points

                    // If classification was enabled:
                    if (face.smilingProbability != null) {
                        val smileProb = face.smilingProbability
                    }
                    if (face.rightEyeOpenProbability != null) {
                        val rightEyeOpenProb = face.rightEyeOpenProbability
                    }

                    // If face tracking was enabled:
                    if (face.trackingId != null) {
                        val id = face.trackingId
                    }
                }
            }
            .addOnFailureListener { e ->
                // Task failed with an exception
                Log.i("error", e.toString())
            }

        var encodingStr = bitmapToString(inputFace)
        try {
            val bytes = py.getModule("skin_predict").callAttr("cut_cheek", encodingStr)
                .toJava(ByteArray::class.java)
            cheekBitMap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }catch (e: Exception) {
            Log.i("error", e.toString())
        }
    }

    fun predictClass(byolArray:FloatArray, py:Python): Int? {
        var classIndex = 0
        try {
            classIndex = py.getModule("skin_predict").callAttr("predict", byolArray)
                .toJava(Int::class.java)
        }catch (e: Exception) {
            Log.i("error", e.toString())
        }
        return classIndex
    }

    // Given an input image, return the estimated score.
    // Note: This is a suspended function, and will run within a CoroutineScope.
    suspend fun predictSkin(image: Bitmap, py:Python) = withContext( Dispatchers.Main ) {
        // 1.切割图像
        cutImage(image, py)
        // 2.生成byol特征
        val scoreOutputArray = Array(1) { FloatArray(512) }
        if (::cheekBitMap.isInitialized) {
            // Input image tensor shape -> [ 1 , 80 , 96 , 3 ]
            val tensorInputImage = TensorImage.fromBitmap(cheekBitMap)
            // Output tensor shape -> [ 1 , 1 ]
            val processedImageBuffer = inputImageProcessor.process(tensorInputImage).buffer
            // Cannot copy to a TensorFlowLite tensor (serving_default_input:0) with 92160 bytes from a Java Buffer with 1080000 bytes.
            // Cannot copy from a TensorFlowLite tensor (PartitionedCall:0) with shape [1, 512] to a Java object with shape [1]
            interpreter?.run(
                processedImageBuffer,
                scoreOutputArray
            )
            // Log.i("Score", scoreOutputArray[0].toString())
        }
        // 3.调用lda
        return@withContext predictClass(scoreOutputArray[0], py)
    }
}



