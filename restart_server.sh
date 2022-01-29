#!/bin/bash
cd /root/python/face/beauty/
ps -ef|grep predict_server|awk '{print $2}'|xargs kill -9
echo "restart server"
nohup /root/anaconda3/envs/face/bin/python predict_server.py > logs/server.log 2>&1 &