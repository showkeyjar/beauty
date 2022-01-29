package com.ml.projects.beautydetection

import android.util.Log
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider.getApplicationContext
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.ml.projects.beautydetection.db.AppDatabase
import com.ml.projects.beautydetection.db.SolutionDao
import org.junit.Assert.assertEquals
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.IOException

/**
 * This is not meant to be a full set of tests. For simplicity, most of your samples do not
 * include tests. However, when building the Room, it is helpful to make sure it works before
 * adding the UI.
 * 参考: https://www.geeksforgeeks.org/testing-room-database-in-android-using-junit/
 * 参考：https://discuss.gradle.org/t/getting-unresolved-reference-error-when-the-reference-can-be-resolved/38994
 * 解决 找不到 AndroidJUnit4 问题
 */

@RunWith(AndroidJUnit4::class)
class AppDatabaseTest {

    private lateinit var solutionDao: SolutionDao
    private lateinit var db: AppDatabase

    @Before
    fun createDb() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        // Using an in-memory database because the information stored here disappears when the
        // process is killed.
//        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
//            .allowMainThreadQueries().build()
        // fallbackToDestructiveMigration 不做数据迁移，
        db = Room.databaseBuilder(
            context,
            AppDatabase::class.java, "beauty.db"
        ).createFromAsset("beauty.db").fallbackToDestructiveMigration().allowMainThreadQueries().build()
        solutionDao = db.solutionDao()
    }

    @After
    @Throws(IOException::class)
    fun closeDb() {
        db.close()
    }

    @Test
    @Throws(Exception::class)
    fun getAll() {
        val solutions = solutionDao.getAll()
        Log.i("db result", solutions.size.toString())
        assertEquals(solutions.size, 13)
    }

    @Test
    @Throws(Exception::class)
    fun findById() {
        val tonight = solutionDao.findById(1)
        Log.i("db result", tonight?.toString())
        assertEquals(tonight?.id, 1)
    }

    @Test
    @Throws(Exception::class)
    fun findByPart() {
        val tonight = solutionDao.findByPart("mouse")
        Log.i("db result", tonight?.toString())
        assertEquals(tonight?.part, "mouse")
    }
}