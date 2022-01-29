package com.ml.projects.beautydetection.db

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query

@Dao
interface SolutionDao {
    @Query("SELECT * FROM solution")
    fun getAll(): List<Solution>

    @Query("SELECT * FROM solution WHERE id IN (:partIds)")
    fun loadAllByIds(partIds: IntArray): List<Solution>

    @Query("SELECT * FROM solution WHERE id=:partId LIMIT 1")
    fun findById(partId: Int): Solution

    @Query("SELECT * FROM solution WHERE part=:partName LIMIT 1")
    fun findByPart(partName: String): Solution

    @Query("SELECT * FROM solution WHERE part IN (:partNames)")
    fun findAllParts(partNames: List<String>): List<Solution>

    @Insert
    fun insertAll(vararg solutions: Solution)

    @Delete
    fun delete(solution: Solution)
}
