package com.ml.projects.beautydetection.ui.top

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class TopViewModel : ViewModel() {

    private val _text = MutableLiveData<String>().apply {
        value = "This is top Fragment"
    }
    val text: LiveData<String> = _text
}