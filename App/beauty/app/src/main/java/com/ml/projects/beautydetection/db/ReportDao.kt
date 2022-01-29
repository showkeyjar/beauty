package com.ml.projects.beautydetection.db

import androidx.room.*

@Dao
interface ReportDao {
    @Query("SELECT * FROM report")
    fun getAll(): List<Report>

    @Query("SELECT * FROM report WHERE id IN (:reportIds)")
    fun loadAllByIds(reportIds: IntArray): List<Report>

    @Query("SELECT * FROM report WHERE name LIKE :reportName LIMIT 1")
    fun findByName(reportName: String): Report

    @Query("SELECT * FROM report WHERE name IN (:reportNames)")
    fun findAllReports(reportNames: List<String>): List<Report>

    @Insert
    fun insertAll(vararg reports: Report)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertOrUpdate(report: Report)

    @Delete
    fun delete(report: Report)
}
