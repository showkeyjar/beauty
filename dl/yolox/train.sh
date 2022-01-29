#！/bin/bash
# ps -ef|grep "tools/train.py" | awk '{print $2}' | xargs kill -9
nohup python tools/train.py -f yolo_voc_s.py -d 4 -b 64 --fp16 -o -c ../models/yolox_s.pth --cache > train.log &