package com.ml.projects.beautydetection.db

import androidx.annotation.NonNull
import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "solution")
data class Solution(
    @PrimaryKey(autoGenerate = true)
    @NonNull
    @ColumnInfo(name = "id")
    val id: Int=0,

    @ColumnInfo(name = "part") val part: String?,
    @ColumnInfo(name = "part_name") val part_name: String?,
    @ColumnInfo(name = "age") val age: Int?,
    @ColumnInfo(name = "score") val score: Float?,
    @ColumnInfo(name = "skin") val skin: Int?,
    @ColumnInfo(name = "method") val method: String?,
    @ColumnInfo(name = "result") val result: String?
)