python solo-learn/main_linear.py \
    --dataset custom \
    --backbone resnet18 \
    --data_dir /data/soft \
    --train_dir skin \
    --no_labels \
    --max_epochs 100 \
    --gpus 0,1,2,3 \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --num_workers 4 \
    --name general-linear-eval \
    --pretrained_feature_extractor trained_models/byol/offline-w7ez810q/byol-400ep-custom-offline-w7ez810q-ep=399.ckpt \
    --project self-supervised \
