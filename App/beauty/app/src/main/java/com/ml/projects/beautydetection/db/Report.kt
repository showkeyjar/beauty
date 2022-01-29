package com.ml.projects.beautydetection.db

import androidx.annotation.NonNull
import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "report")
data class Report(
    @PrimaryKey(autoGenerate = true)
    @NonNull
    @ColumnInfo(name = "id")
    val id: Int=0,

    @ColumnInfo(name = "name") val name: String?,
    @ColumnInfo(name = "age") val age: Int?,
    @ColumnInfo(name = "score") val score: Float?,
    @ColumnInfo(name = "skin") val skin: Int?,
    @ColumnInfo(name = "content") val content: String?
)