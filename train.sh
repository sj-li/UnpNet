#!/bin/bash

#block(name=pointconv-512, threads=2, memory=5000, gpus=1, hours=1000)
python train.py --dataset data/single_scan --arch_cfg config/arch/PointConv-512.yaml --data_cfg config/labels/semantic-kitti.yaml --log logs/pointconv-512
