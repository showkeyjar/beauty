package com.ml.projects.beautydetection

import android.os.Bundle
import com.google.android.material.bottomnavigation.BottomNavigationView
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupWithNavController
import com.jiang.awesomedownloader.core.AwesomeDownloader
import com.ml.projects.beautydetection.databinding.ActivityMainBinding
import com.umeng.commonsdk.UMConfigure
import com.ml.projects.beautydetection.ui.top.TopFragment


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val navView: BottomNavigationView = binding.navView

        val navController = findNavController(R.id.nav_host_fragment_activity_main)
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        val appBarConfiguration = AppBarConfiguration(
            setOf(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications
            )
        )
        navView.setupWithNavController(navController)

        UMConfigure.preInit(this, "6190a755e0f9bb492b5a4391", "Umeng")
        UMConfigure.init(this, "6190a755e0f9bb492b5a4391", "Umeng", UMConfigure.DEVICE_TYPE_PHONE, "")

        //前台服务模式启动（独立启动，直至服务被kill或关闭）传入能创建服务的ContextWrapper
        // AwesomeDownloader.initWithServiceMode(this)
    }

    override fun onBackPressed() {
        val fragment = supportFragmentManager.findFragmentByTag("TopFragment")
        if (TopFragment::class.java.isInstance(fragment)) {
            if (onBackPressedDispatcher.hasEnabledCallbacks()) {
                onBackPressedDispatcher.onBackPressed()
                return
            }
            super.onBackPressed()
        }
    }

}