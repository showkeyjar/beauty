package com.ml.projects.beautydetection.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase


@Database(entities = [Solution::class, Report::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun solutionDao(): SolutionDao
    abstract fun reportDao(): ReportDao

    companion object {

        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getInstance(context: Context): AppDatabase {
            synchronized(this) {
                var instance = INSTANCE

                if (instance == null) {
                    instance = Room.databaseBuilder(
                        context.applicationContext,
                        AppDatabase::class.java,
                        "beauty.db"
                    ).createFromAsset("beauty.db")
                        .fallbackToDestructiveMigration()
                        .build()
                    INSTANCE = instance
                }
                return instance
            }
        }
    }
}