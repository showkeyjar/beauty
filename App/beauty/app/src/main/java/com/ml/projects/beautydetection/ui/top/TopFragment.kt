package com.ml.projects.beautydetection.ui.top

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.WebSettings
import android.webkit.WebViewClient
import androidx.activity.OnBackPressedCallback
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.ml.projects.beautydetection.databinding.FragmentTopBinding

class TopFragment : Fragment(){

    private lateinit var topViewModel: TopViewModel
    private var _binding: FragmentTopBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        topViewModel =
            ViewModelProvider(this).get(TopViewModel::class.java)

        _binding = FragmentTopBinding.inflate(inflater, container, false)
        val root: View = binding.root

//        val textView: TextView = binding.textTop
//        topViewModel.text.observe(viewLifecycleOwner, Observer {
//            textView.text = it
//        })

        var webView = binding.webView
        // WebViewClient allows you to handle
        // onPageFinished and override Url loading.
        webView.webViewClient = WebViewClient()

        // this will load the url of the website
        webView.loadUrl("http://www.1mei.fit/")

        // this will enable the javascript settings
        webView.settings.javaScriptEnabled = true

        webView.settings.domStorageEnabled = true

        webView.settings.databaseEnabled = true

        // if you want to enable zoom feature
        webView.settings.setSupportZoom(true)

        webView.settings.setAppCacheEnabled(false)

        webView.settings.cacheMode = WebSettings.LOAD_NO_CACHE


        val callback = object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if(webView.canGoBack()){
                    webView.goBack()
                } else {
                    isEnabled = false
                    requireActivity().onBackPressed()
                }
            }
        }
        requireActivity().onBackPressedDispatcher.addCallback(callback)

        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}