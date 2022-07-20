## What it is

This repository is a pytorch version multi-gpu KP_Pyramid for point cloud data processing (some codes haven't been released now). We will release a clean version of code later, you can refer to a new version for training then.

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

## Cite
<pre>
@inproceedings{nie2022pyramid,
  title={Pyramid Architecture for Multi-Scale Processing in Point Cloud Segmentation},
  author={Nie, Dong and Lan, Rui and Wang, Ling and Ren, Xiaofeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17284--17294},
  year={2022}
}
</pre>

Our work uses part of codes from https://github.com/HuguesTHOMAS/KPConv-PyTorch. Thanks for the great work!

