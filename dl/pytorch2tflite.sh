# pytorch模型转tflite
# https://github.com/omerferhatt/torch2tflite
cp /mnt/data/soft/skin/cheek/AF9right_cheek.jpg test.jpg
python converter.py --torch-path byol.pt --tflite-path byol_skin.tflite --target-shape 80 96 3 --sample-file test.jpg
