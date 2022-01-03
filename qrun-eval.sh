#block(name=pointconv_b1-eval, threads=10, memory=10000, gpus=1, hours=1000)
python infer.py --dataset data/single_scan/ --model PointConv_B1 --arch_cfg config/arch/PointConv_B1-1024.yaml  --data_cfg config/labels/semantic-kitti.yaml --checkpoint logs/pointconv_b1-1024/best_val-epoch-0031.path --log predictions-3
#&& python evaluate_iou.py -d data/single_scan/ -p predictions-2
