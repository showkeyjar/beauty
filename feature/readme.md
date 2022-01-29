# 特征抽取

scikit-image 的 multi-block lbp 方法已完成，可以尝试下


# 传统方案(v1.0)

## 皮肤检测

deepgaze 皮肤检测

The Histogram Backprojection Algorithm

Human Skin Detection Using RGB, HSV And Ycbcr Color Models

https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf

皮肤颜色的均值、方差等

初期以lbph特征代替，关于lbph的详细说明：
[Local Binary Patterns Histograms](https://blog.csdn.net/Zachary_Co/article/details/78807627)


## 整体特征

尝试使用 keras_vggface：

http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

https://github.com/bknyaz/beauty_vision
