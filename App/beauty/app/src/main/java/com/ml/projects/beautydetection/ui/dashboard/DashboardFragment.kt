package com.ml.projects.beautydetection.ui.dashboard

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProvider
import androidx.room.Room
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.ml.projects.beautydetection.databinding.FragmentDashboardBinding
import com.ml.projects.beautydetection.db.AppDatabase
import com.ml.projects.beautydetection.db.Report
import com.github.mikephil.charting.data.LineDataSet
import de.codecrafters.tableview.TableView
import de.codecrafters.tableview.model.TableColumnWeightModel
import de.codecrafters.tableview.toolkit.SimpleTableDataAdapter


class DashboardFragment : Fragment() {
    private lateinit var db: AppDatabase
    private lateinit var dashboardViewModel: DashboardViewModel
    private var _binding: FragmentDashboardBinding? = null

    // 参考：
    // https://github.com/AAChartModel/AAChartCore-Kotlin 注意 AAChart 概念混乱不适合使用
    // https://github.com/PhilJay/MPAndroidChart
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

        dashboardViewModel =
            ViewModelProvider(this).get(DashboardViewModel::class.java)

        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textDashboard
        dashboardViewModel.text.observe(viewLifecycleOwner, Observer {
            textView.text = it
        })

        var textInfo: TextView = binding.textInfo
        var textReport: TextView = binding.textReport
        textReport.text = "颜值报告"

        var tableView: TableView<Array<String?>> = binding.myReport as TableView<Array<String?>>

        val scoreChartView = binding.scoreChart
        var ageChartView = binding.ageChart

        var reports:List<Report> = db.reportDao().getAll()
        Log.i("db", "find report records:" + reports.size.toString())
        if(reports.size==0){
            scoreChartView.visibility = View.INVISIBLE
            ageChartView.visibility = View.INVISIBLE
            textInfo.text = "暂无数据"
            textInfo.visibility = View.VISIBLE
        }else {
            var scores = ArrayList<Entry>()
            var ages = ArrayList<Entry>()

            var data = ArrayList<Array<String?>>()

            reports.forEachIndexed { index, it ->
                it.score?.let { it2 -> Entry(index.toFloat(), it2) }?.let { it3 ->
                    scores.add(
                        it3
                    )
                }

                it.age?.let { it2 -> Entry(index.toFloat(), it2.toFloat()) }?.let { it3 ->
                    ages.add(
                        it3
                    )
                }
                it.content?.let { it1 -> data.add(arrayOf(it1)) }

            }
            Log.i("chart", arrayOf(scores).toString())
            Log.i("chart", arrayOf(ages).toString())

            val scoreSet = LineDataSet(scores, "Score")
            var scoreData = LineData(scoreSet)
            scoreChartView.setData(scoreData)
            scoreChartView.description.text = "颜值评分变化趋势"

            val ageSet = LineDataSet(ages, "Score")
            var ageData = LineData(ageSet)
            ageChartView.setData(ageData)
            ageChartView.description.text = "预测年龄变化趋势"


            tableView.setColumnCount(2)
            val columnModel = TableColumnWeightModel(1)
            columnModel.setColumnWeight(0, 2)
            tableView.setColumnModel(columnModel)
            tableView.setDataAdapter(SimpleTableDataAdapter(requireContext(), data))
        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        try{
            db.close()
        }catch (e: Exception){}
    }
}