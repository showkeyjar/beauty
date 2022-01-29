package com.ml.projects.beautydetection.ui.home

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

/*
ViewModel 你妹！ 对项目开发者越来越不友好！
 */
class HomeViewModel : ViewModel() {

    private val _text = MutableLiveData<String>().apply {
        value = "系统不会上传您的图像，所有推断都在本地计算(友盟SDK收集APP使用信息用于统计分析)"
    }
    val text: LiveData<String> = _text
}