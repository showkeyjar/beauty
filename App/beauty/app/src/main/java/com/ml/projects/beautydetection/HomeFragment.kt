package com.ml.projects.beautydetection

import android.app.Activity.RESULT_OK
import android.app.ProgressDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import android.media.ExifInterface
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import androidx.room.Room
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.projects.beautydetection.databinding.FragmentHomeBinding
import com.ml.projects.beautydetection.db.AppDatabase
import com.ml.projects.beautydetection.db.Report
import com.ml.projects.beautydetection.ui.home.HomeViewModel
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.math.RoundingMode
import java.text.DecimalFormat
import kotlin.collections.ArrayList
import kotlin.math.floor

class HomeFragment : Fragment() {

    private lateinit var progressDialog : ProgressDialog

    // Initialize the MLKit FaceDetector
    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
        .build()
    private val firebaseFaceDetector = FaceDetection.getClient(realTimeOpts)

    // CoroutineScope in which we'll run our coroutines.
    private val coroutineScope = CoroutineScope( Dispatchers.Main )

    private lateinit var sampleImageView : ImageView
    private lateinit var ageOutputTextView : TextView
    private lateinit var beautyOutputTextView: TextView
    private lateinit var genderOutputTextView : TextView
    private lateinit var skinOutputTextView: TextView
    // private lateinit var inferenceSpeedTextView : TextView
    private lateinit var resultsLayout : ConstraintLayout

    // For reading the full-sized picture
    private val REQUEST_IMAGE_CAPTURE = 101
    private val REQUEST_IMAGE_SELECT = 102
    private lateinit var currentPhotoPath : String

    private lateinit var reportFace: Face
    private lateinit var reportBitMap: Bitmap
    private var facePartName:Map<String, String> = mapOf(
        "nose_bridge" to "鼻梁", "nose_tip" to "鼻尖",
        "left_cheek" to "左脸蛋", "right_cheek" to "右脸蛋",
        "left_eyebrow" to "左眉", "right_eyebrow" to "右眉",
        "left_eye" to "左眼", "right_eye" to "右眼",
        "upper_lip" to "上唇", "lower_lip" to "下唇",
        "left_forehead" to "左额", "right_forehead" to "右额",
        "mouse" to "嘴巴"
    )
    private var skinClassNames:Map<Int, String> = mapOf(
        45 to "粗糙黄皮肤", 101 to "粗糙白皮肤",
        118 to "光滑黄皮肤", 129 to "光滑白皮肤"
    )
    private val shift = 5

    // Boolean values to check for NNAPI and Gpu Delegates
    private var isModelInit : Boolean = false
    private var useNNApi : Boolean = false
    private var useGpu : Boolean = false
    private val compatList = CompatibilityList()
    // Default model filename
    private var modelFilename = arrayOf( "model_age.tflite", "model_gender.tflite", "python/model_beauty_q_v2.tflite", "python/byol_skin.tflite" )
    // TFLite interpreters for both the models
    lateinit var ageModelInterpreter: Interpreter
    lateinit var beautyModelInterpreter: Interpreter
    lateinit var genderModelInterpreter: Interpreter
    lateinit var skinModelInterpreter: Interpreter

    lateinit var ageEstimationModel: AgeEstimationModel
    lateinit var beautyEstimationModel: BeautyEstimationModel
    lateinit var genderClassificationModel: GenderClassificationModel
    lateinit var skinEstimationModel: SkinEstimationModel
    lateinit var reportRecord: Map<String, String>
    lateinit var saveScore: Button

    private lateinit var homeViewModel: HomeViewModel
    private lateinit var db: AppDatabase
    private lateinit var py: Python
    private var _binding: FragmentHomeBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    fun init_db(){
        db = Room.databaseBuilder(
            requireActivity().getApplicationContext(),
            AppDatabase::class.java, "beauty.db"
        ).createFromAsset("beauty.db").fallbackToDestructiveMigration().allowMainThreadQueries().build()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        init_db()
        homeViewModel =
            ViewModelProvider(this).get(HomeViewModel::class.java)

        _binding = FragmentHomeBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textHome
        sampleImageView = binding.sampleInputImageview
        // inferenceSpeedTextView = binding.inferenceSpeedTextView
        ageOutputTextView = binding.ageOutputTextView
        beautyOutputTextView = binding.beautyOutputTextview
        genderOutputTextView = binding.genderOutputTextview
        skinOutputTextView = binding.skinOutputTextview
        resultsLayout = binding.resultsLayout

        //拍照
        val openCameraButton: Button = binding.openCameraButton
        openCameraButton.setOnClickListener { openCamera() }
        //相册
        var selectImageButton: Button = binding.selectImageButton
        selectImageButton.setOnClickListener { selectImage() }
        //颜值分析
        var faceButton: Button = binding.faceButton
        faceButton.setOnClickListener { faceReport() }
        //皮肤分析
//        var skinButton: Button = binding.skinButton
//        skinButton.setOnClickListener{ skinReport() }
        //保存结果
        saveScore = binding.saveScore
        saveScore.setOnClickListener { saveFaceScore() }
        //参与排行
        var faceTop: Button = binding.faceTop
        faceTop.setOnClickListener { joinTop() }

        homeViewModel.text.observe(viewLifecycleOwner, Observer {
            textView.text = it
        })

        progressDialog = ProgressDialog(requireActivity())
        progressDialog.setCancelable( false )
        progressDialog.setMessage( "搜索人脸 ...")

        if(!isModelInit){
            modelInit();
            isModelInit = true;
        }

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        try{
            ageModelInterpreter.close()
        }catch (e: Exception){}
        try{
            genderModelInterpreter.close()
        }catch (e: Exception){}
        try{
            beautyModelInterpreter.close()
        }catch (e: Exception){}
        try{
            db.close()
        }catch (e: Exception){}
    }

    // Suspending function to initialize the TFLite interpreters.
    private suspend fun initModels(options: Interpreter.Options) = withContext( Dispatchers.Default ) {
        ageModelInterpreter = Interpreter(FileUtil.loadMappedFile( requireActivity().getApplicationContext() , modelFilename[0]), options )
        genderModelInterpreter = Interpreter(FileUtil.loadMappedFile( requireActivity().getApplicationContext() , modelFilename[1]), options )
        beautyModelInterpreter = Interpreter(FileUtil.loadMappedFile( requireActivity().getApplicationContext() , modelFilename[2]), options )
        skinModelInterpreter = Interpreter(FileUtil.loadMappedFile( requireActivity().getApplicationContext() , modelFilename[3]), options )
        withContext( Dispatchers.Main ){
            ageEstimationModel = AgeEstimationModel().apply {
                interpreter = ageModelInterpreter
            }
            genderClassificationModel = GenderClassificationModel().apply {
                interpreter = genderModelInterpreter
            }
            beautyEstimationModel = BeautyEstimationModel().apply {
                interpreter = beautyModelInterpreter
            }
            skinEstimationModel = SkinEstimationModel().apply {
                interpreter = skinModelInterpreter
            }
            // Notify the user once the models have been initialized.
            Toast.makeText( requireActivity().getApplicationContext() , "模型初始化..." , Toast.LENGTH_LONG ).show()
        }
    }


    private fun modelInit() {
        //自动选择最合适的推理技术
        if ( Build.VERSION.SDK_INT >= Build.VERSION_CODES.P ) {
            useNNApi = true
        }
        if ( compatList.isDelegateSupportedOnThisDevice ){
            useGpu = true
        }

        val options = Interpreter.Options().apply {
            if ( useGpu ) {
                addDelegate(GpuDelegate( compatList.bestOptionsForThisDevice ) )
            }else if ( useNNApi ) {
                addDelegate(NnApiDelegate())
            }
        }
        // Initialize the models in a coroutine.
        coroutineScope.launch {
            initModels(options)
        }

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(requireActivity().getApplicationContext()))
            Log.i("Python","python start")
        }
        if(!::py.isInitialized){
            py = Python.getInstance()
        }
    }

    private fun openCamera() {
        dispatchTakePictureIntent()
    }

    private fun selectImage() {
        dispatchSelectPictureIntent()
    }

    private fun faceReport() {
        showFaceReportDialog()
    }

    private fun skinReport(){
        // 皮肤检测
    }

    private fun saveFaceScore(){
        // 将得分存储到数据库
        Log.i("saveDB", "save score to report")
        if(::reportRecord.isInitialized){
            var report:Report = Report(
                name = reportRecord["name"],
                age = reportRecord["age"]?.toInt(),
                score = reportRecord["score"]?.toFloat(),
                skin = reportRecord["skin"]?.toInt(),
                content = ""
            )
            db.reportDao().insertOrUpdate(report)
            Toast.makeText(requireActivity().getApplicationContext(), "评测结果已保存", Toast.LENGTH_SHORT).show()
            saveScore.isClickable = false
        }
    }

    private fun joinTop(){
        // todo 参与颜值排行
        Log.i("joinTop", "join user info to top")
    }

    suspend fun genFaceReport(py: Python, reportImageView: ImageView, textViewFacePart:TextView) = coroutineScope {
        var bones = ArrayList<ArrayList<Array<Float>>>()
        var do_face_part = async(Dispatchers.IO){
            // output FaceContour score
            // https://developers.google.com/ml-kit/vision/face-detection/face-detection-concepts#contours
            Log.i("contour", "get face contour parts")
            var bbox = reportFace.boundingBox
            var contours = reportFace.allContours
            for(contour in contours){
                var c_points = ArrayList<Array<Float>>()
                for(p in contour.points){
                    var c_point = arrayOf(p.x - (bbox.left - 0 * shift), p.y - (bbox.top + shift))
                    c_points.add(c_point)
                }
                Log.i(contour.faceContourType.toString(), "contour:[" + contour.faceContourType.toString() + "]" + c_points.toString())
                bones.add(c_points)
            }
        }
        var finish_score = false
        val do_face_score = async(Dispatchers.IO) {
            val inputFace = (sampleImageView.getDrawable() as BitmapDrawable).bitmap
            var encodingStr = bitmapToString(inputFace)
            try {
                val bytes = py.getModule("report_lite").callAttr("gen_result", encodingStr)
                    .toJava(ByteArray::class.java)
                reportBitMap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }catch (e: Exception) {
                Log.i("error", e.toString())
                textViewFacePart.text = "分析错误，请换张图片试试!"
            }finally {
                finish_score = true
            }
        }
        do_face_part.await()
        progressDialog.setMessage( "取得人脸部位 ...")
        var do_progress = async(Dispatchers.IO){
            var wait_step = 0
            while(!finish_score){
                Thread.sleep(3000)
                progressDialog.setMessage("计算部位得分" + ".".repeat(wait_step))
                wait_step++
            }
        }
        do_face_score.await()
        if (::reportBitMap.isInitialized) {
            reportImageView.setImageBitmap(reportBitMap)
            do_progress.await()
            var parts = ArrayList<String>()
            progressDialog.setMessage("检测缺陷 ...")
            var do_part_score = async(Dispatchers.IO) {
                var reportStr = bitmapToString(reportBitMap)
                val array: Array<Array<Array<Float>>?> = arrayOfNulls(bones.size)
                for (i in 0 until bones.size) {
                    val row: ArrayList<Array<Float>> = bones.get(i)
                    array[i] = row.toArray(arrayOfNulls(row.size))
                }
                val scores: Map<PyObject, PyObject> =
                    py.getModule("part_mlkit").callAttr("get_contour_values", reportStr, array)
                        .asMap()
                Log.i("contour", "face_part scores:" + scores.toString())
                for ((key, value) in scores) {
                    if (value.toFloat() > 0.0) {
                        facePartName.get(key.toString())?.let { parts.add(it) }
                    }
                }
            }
            do_part_score.await()
            textViewFacePart.text = parts.joinToString(separator = ",")
        }
        progressDialog.dismiss()
    }

    private fun showFaceReportDialog(){
        val alertDialogBuilder = AlertDialog.Builder(requireContext())
        alertDialogBuilder.setCancelable( false )
        // 人脸分析
        val faceView = layoutInflater.inflate( R.layout.beauty_report, null )
        var genReportButton : Button = faceView.findViewById( R.id.report_button)
        val closeButton : Button = faceView.findViewById( R.id.close_button )
        var reportImageView : ImageView = faceView.findViewById( R.id.beauty_report_imageview )
        var textViewFacePart: TextView = faceView.findViewById( R.id.textViewFacePart )

        alertDialogBuilder.setView(faceView)
        val dialog = alertDialogBuilder.create()
        dialog.show()

        genReportButton.setOnClickListener{
            val intent = Intent(getActivity(), BeautyReportActivity::class.java)
            intent.putExtra("face_part", textViewFacePart.text)
            intent.putExtra("face_name", reportRecord["name"])
            intent.putExtra("face_age", reportRecord["age"].toString())
            intent.putExtra("face_score", reportRecord["score"].toString())
            intent.putExtra("face_skin", reportRecord["skin"].toString())
            startActivity(intent)
        }

        closeButton.setOnClickListener {
            dialog.dismiss()
        }

        progressDialog.setMessage( "分析人脸 ...")
        progressDialog.show()
        if(reportFace != null){
            CoroutineScope(Dispatchers.Main).launch {
                genFaceReport(py, reportImageView, textViewFacePart)
            }
        }
        else{
            progressDialog.dismiss()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        // If the user opened the camera
        progressDialog.setMessage( "搜索人脸 ...")
        if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_CAPTURE ) {
            // Get the full-sized Bitmap from `currentPhotoPath`.
            var bitmap = BitmapFactory.decodeFile( currentPhotoPath )
            val exifInterface = ExifInterface( currentPhotoPath )
            bitmap =
                when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION , ExifInterface.ORIENTATION_UNDEFINED )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap( bitmap , 90f )
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap( bitmap , 180f )
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap( bitmap , 270f )
                    else -> bitmap
                }
            progressDialog.show()
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
        // if the user selected an image from the gallery
        else if ( resultCode == RESULT_OK && requestCode == REQUEST_IMAGE_SELECT ) {
            val inputStream = requireActivity().contentResolver.openInputStream( data?.data!! )
            val bitmap = BitmapFactory.decodeStream( inputStream )
            inputStream?.close()
            progressDialog.show()
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
    }

    private fun mathFloor(number: Double, size:Int=2): String {
        var formatStr = "0." + "#".repeat(size)
        val format = DecimalFormat(formatStr)
        //未保留小数的舍弃规则，RoundingMode.FLOOR表示直接舍弃
        format.roundingMode = RoundingMode.FLOOR
        return format.format(number)
    }

    private fun detectFaces(image: Bitmap) {
        val inputImage = InputImage.fromBitmap(image, 0)
        // Pass the clicked picture to MLKit's FaceDetector.
        firebaseFaceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if ( faces.size != 0 ) {
                    reportFace = faces[0]
                    var inputFace = cropToBBox(image, reportFace.boundingBox)
                    // Set the cropped Bitmap into sampleImageView.
                    sampleImageView.setImageBitmap(inputFace)
                    // Launch a coroutine
                    coroutineScope.launch {
                        // Predict the age and the gender.
                        val age_pred = ageEstimationModel.predictAge(inputFace)
                        var age = floor(age_pred.toDouble()).toInt().toString()
                        val gender = genderClassificationModel.predictGender(inputFace)
                        var beauty = beautyEstimationModel.predictScore(inputFace)
                        // 先对人脸进行分割，切分出cheek作为输入
                        var skin = 0
                        if(::py.isInitialized) {
                            skin = skinEstimationModel.predictSkin(inputFace, py)!!
                        }
                        val currentTs = floor(System.currentTimeMillis().toDouble()).toString()
                        reportRecord= mapOf("name" to currentTs,
                            "age" to age, "gender" to gender.toString(),
                            "score" to beauty.toString(),
                            "skin" to skin.toString() )

                        if(::saveScore.isInitialized){
                            saveScore.isClickable = true
                        }

                        // Show the final output to the user.
                        ageOutputTextView.text = age
                        genderOutputTextView.text = if ( gender[ 0 ] > gender[ 1 ] ) { "男" } else { "女" }
                        beautyOutputTextView.text = mathFloor( beauty.toDouble() )
                        skinOutputTextView.text = skinClassNames[skin]
                        resultsLayout.visibility = View.VISIBLE
                        progressDialog.dismiss()
                    }
                }
                else {
                    // Show a dialog to the user when no faces were detected.
                    progressDialog.dismiss()
                    val dialog = AlertDialog.Builder( requireContext() ).apply {
                        // title = "未检测到人脸"
                        setMessage( "照片上未发现人脸，请换张图片再试试 " )
                        setPositiveButton( "OK") { dialog, which ->
                            dialog.dismiss()
                        }
                        setCancelable( false )
                        create()
                    }
                    dialog.show()
                }
            }
            .addOnFailureListener {
                // Show a dialog to the user when no faces were detected.
                progressDialog.dismiss()
                val dialog = AlertDialog.Builder( requireContext() ).apply {
                    // title = "图像读取错误"
                    setMessage( "请检查图像格式或更换一张图像再重试. " )
                    setPositiveButton( "OK") { dialog, which ->
                        dialog.dismiss()
                    }
                    setCancelable( false )
                    create()
                }
                dialog.show()
            }
    }


    private fun cropToBBox(image: Bitmap, bbox: Rect) : Bitmap {
        return Bitmap.createBitmap(
            image,
            bbox.left - 0 * shift,
            bbox.top + shift,
            bbox.width() + 0 * shift,
            bbox.height() + 0 * shift
        )
    }

    // Create a temporary file, for storing the full-sized picture taken by the user.
    private fun createImageFile() : File {
        val imagesDir = requireActivity().getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", ".jpg", imagesDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    // Dispatch an Intent which opens the gallery application for the user.
    private fun dispatchSelectPictureIntent() {
        val selectPictureIntent = Intent( Intent.ACTION_OPEN_DOCUMENT ).apply {
            type = "image/*"
            addCategory( Intent.CATEGORY_OPENABLE )
        }
        startActivityForResult( selectPictureIntent , REQUEST_IMAGE_SELECT )
    }

    // Dispatch an Intent which opens the camera application for the user.
    // The code is from -> https://developer.android.com/training/camera/photobasics#TaskPath
    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent( MediaStore.ACTION_IMAGE_CAPTURE )
        if ( takePictureIntent.resolveActivity( requireActivity().packageManager ) != null ) {
            val photoFile: File? = try {
                createImageFile()
            }
            catch (ex: IOException) {
                null
            }
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    requireContext(),
                    "com.ml.projects.beautydetection", it
                )
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }


    private fun rotateBitmap(original: Bitmap, degrees: Float): Bitmap? {
        val matrix = Matrix()
        matrix.preRotate(degrees)
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }

    private fun bitmapToString(inputFace: Bitmap):String{
        val stream = ByteArrayOutputStream()
        inputFace.compress(Bitmap.CompressFormat.PNG, 90, stream)
        val py_image = stream.toByteArray()
        val encodedString: String = android.util.Base64.encodeToString(py_image, android.util.Base64.DEFAULT)
        return encodedString
    }
}