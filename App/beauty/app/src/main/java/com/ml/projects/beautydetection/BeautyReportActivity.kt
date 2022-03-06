package com.ml.projects.beautydetection

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.room.Room
import com.ml.projects.beautydetection.db.AppDatabase
import com.ml.projects.beautydetection.db.Report
import com.ml.projects.beautydetection.db.Solution
import de.codecrafters.tableview.TableView
import de.codecrafters.tableview.model.TableColumnWeightModel
import de.codecrafters.tableview.toolkit.SimpleTableDataAdapter


class BeautyReportActivity : AppCompatActivity(){
    private lateinit var saveButton: Button
    private lateinit var reportRecord: Report
    private lateinit var report: Report
    private lateinit var db: AppDatabase
    private lateinit var tableView : TableView<Array<String?>>
    private var facePartName:Map<String, String> = mapOf(
        "鼻梁" to "nose_bridge", "鼻尖" to "nose_tip",
        "左脸蛋" to "left_cheek", "右脸蛋" to "right_cheek",
        "左眉" to "left_eyebrow", "右眉" to "right_eyebrow",
        "左眼" to "left_eye", "右眼" to "right_eye",
        "上唇" to "upper_lip", "下唇" to "lower_lip",
        "左额" to "left_forehead", "右额" to "right_forehead",
        "嘴巴" to "mouse"
    )

    fun init_db(){
        db = Room.databaseBuilder(
            applicationContext,
            AppDatabase::class.java, "beauty.db"
        ).createFromAsset("beauty.db").fallbackToDestructiveMigration().allowMainThreadQueries().build()
    }

    fun search_solutions(parts: List<String>):List<Solution>{
        return db.solutionDao().findAllParts(parts)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.gen_report)
        val faceParts = intent.getStringExtra("face_part").toString()
        var name = intent.getStringExtra("face_name").toString()
        var age = intent.getStringExtra("face_age").toString()
        var score = intent.getStringExtra("face_score").toString()
        var skin = intent.getStringExtra("face_skin").toString()
        init_db()
        reportRecord = db.reportDao().findByName(name)
        saveButton = findViewById(R.id.save_button)
        saveButton.isClickable = true
        tableView = findViewById(R.id.genTableView1)
        tableView.setColumnCount(4)

        val parts = faceParts.split(",")
        Log.i("part", "get parts:" + parts.toString())
        var part_keys = ArrayList<String>()
        for (part in parts){
            facePartName.get(part)?.let { part_keys.add(it) }
        }
        //var solutions:List<Solution> = db.solutionDao().findAllParts("'" + parts.joinToString("','") + "'")
        var solutions:List<Solution> = search_solutions(part_keys)
        Log.i("sqlite", "find solutions " + solutions.size.toString())
        var data = arrayOfNulls<Array<String?>>(solutions.size)
        var content = ""
        for (i in 0 until solutions.size){
            var s = solutions[i]
            var row = arrayOf(s.part, s.part_name, s.method, s.result)
            data[i] = row
            content += row.toString()
        }
        if(::reportRecord.isInitialized){
            report = Report(id = reportRecord.id,
                name = reportRecord.name,
                age = reportRecord.age,
                score = reportRecord.score,
                skin = reportRecord.skin,
                content = content
            )
        }else{
            report = Report(
                name = name,
                age = age.toInt(),
                score = score.toFloat(),
                skin = skin.toInt(),
                content = content
            )
        }
        Log.i("sqlite", "get data " + data.size.toString())
        val columnModel = TableColumnWeightModel(4)
        columnModel.setColumnWeight(1, 2)
        columnModel.setColumnWeight(2, 2)
        tableView.setColumnModel(columnModel)
        tableView.setDataAdapter(SimpleTableDataAdapter(this, data))
    }

    fun saveResult(v : View){
        Log.i("save", "save result")
        if(::report.isInitialized){
            db.reportDao().insertOrUpdate(report)
            Toast.makeText(applicationContext, "评测报告已保存", Toast.LENGTH_SHORT).show()
            saveButton.isClickable = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try{
            db.close()
        }catch (e: Exception){}
    }
}
