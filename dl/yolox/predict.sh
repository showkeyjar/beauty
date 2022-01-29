#！/bin/bash
# ps -ef|grep "tools/eval.py" | awk '{print $2}' | xargs kill -9
nohup python tools/eval.py -n yolox-s -c ../models/last_epoch_ckpt.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse > predict.log &