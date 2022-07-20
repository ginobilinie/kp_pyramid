## What it is

This repository is a pytorch version multi-gpu KP_Pyramid for point cloud data processing.

## How to train

For multi-gpu training

```
nohup python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM train_dist.py --network pyramid_v2_deform --test-area-idx 0 >log_train_pyramid_v2_deform_s3dis_area0_0818.txt 2>&1 &
```

For single-gpu training

```
nohup python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM train_dist.py --network pyramid_v2_deform --test-area-idx 0 >log_train_pyramid_v2_deform_s3dis_area0_0818.txt 2>&1 &
```

## How to test

```
nohup python test_full_cloud.py result/kp_pyramid_v2_lr0p01_area0/ --dataset S3DIS --test-area-idx 0 --network pyramid_v2_deform --num-votes 50 >log_test_pyramid_v2_deform_s3dis_area0_0818_e500.txt 2>&1 &
```
