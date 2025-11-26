# Downstream Tasks
This section describes the downstream tasks that can be performed using the DINO-Foresight model. The following tasks are supported:
1. Semantic Segmentation
2. Depth Estimation
3. Surface Normal Estimation

## Training DPT Heads
In order to train DPT heads for semantic segmentation with default settings use:
```bash
python train_dpt_head.py --img_size 448,896 --modality "segm" --batch_size 16 --max_epochs 100 \
    --warmup_p 0.1 --num_gpus 8 --precision "16-mixed" --data_path "path/to/cityscapes" \
    --lr_base 1e-4 --optimizer "adamw" --weight_decay 1e-4 --scheduler "poly" --eval_freq 1  \
    --feature_extractor "dino" --num_classes 19 --dlayers 2,5,8,11 --dpt_out_channels 128,256,512,512 --use_bn \
    --pca_ckpt "path/to/pca/dinov2_pca_448_l[2_5_8_11]_1152.pth" \
    --dst_path "/logs/dino_segm_dpt_pca" \
```
In order to train DPT heads for depth estimation with default settings use:
```bash
python train_dpt_head.py --img_size 448,896 --modality "depth" --batch_size 16 --max_epochs 100 \      
      --warmup_p 0.1 --num_gpus 8 --precision "16-mixed" ---data_path "path/to/cityscapes" \
      --lr_base 1e-4 --optimizer "adamw" --weight_decay 1e-4 --scheduler "cosine" --eval_freq 1  \
      --feature_extractor "dino" --num_classes 256 --dlayers 2,5,8,11 --dpt_out_channels 128,256,512,512 --use_bn \
      --pca_ckpt "path/to/pca/pca_448_l[2_5_8_11]_1152.pth" \
      --dst_path "logs/dino_depth_dpt_pca" \
```
In order to train DPT heads for surface normal estimation with default settings use:
```bash
python train_dpt_head.py --img_size 448,896 --modality "surface_normals" --batch_size 16 --max_epochs 100 \
      --warmup_p 0.1 --num_gpus 8 --precision 16-mixed --data_path "path/to/cityscapes" \
      --lr_base 1e-4 --optimizer "adamw" --weight_decay 1e-4 --scheduler "poly" --eval_freq 1  \
      --feature_extractor "dino" --num_classes 3 --dlayers 2,5,8,11 --dpt_out_channels 128,256,512,512 --use_bn \
      --pca_ckpt "path/to/pca/pca_448_l[2_5_8_11]_1152.pth"  \
      --dst_path "/logs/dino_surfnorm_dpt" \
```
Key Arguments:
- `--data_path`: Path to Cityscapes dataset (default: "/home/ubuntu/cityscapes")
- `--img_size`: Input image dimensions as tuple (default: (448,896))
- `--batch_size`: Batch size for training (default: 16)
- `--feature_extractor`: Type of backbone network (choices: ['dino', 'eva2-clip', 'sam'], default: 'dino')
- `--num_classes`: Number of output classes:
  - 19 for segmentation (Classification)
  - 256 for depth (Classification)
  - 3 for surface normals (Regression)
- `--dlayers`: Which DINO layers to use, zero-indexed (default: [2,5,8,11])
- `--pca_ckpt`: Path to PCA checkpoint for feature reconstruction (optional)
- `--lr_base`: Base learning rate (default: 1e-4)
- `--max_epochs`: Number of training epochs (default: 100)
- `--optimizer`: Optimization algorithm (choices: ["adam", "adamw"], default: "adamw")
- `--scheduler`: Learning rate scheduler (choices: ["cosine", "poly"], default: "poly")

## Checkpoints 
You can download the pre-trained models from the following links:

|Task|Checkpoint|$mIoU$|$MO\_mIoU$|
| - | - | - | - |
| Semantic Segmentation| [Download](https://huggingface.co/Sta8is/DINO-Foresight/blob/main/head_segm_pca1152.ckpt) | 77 | 77.4 |

or via command line:
```bash
wget https://huggingface.co/Sta8is/DINO-Foresight/resolve/main/head_segm_pca1152.ckpt
```

| Task | Checkpoint |$\delta_1$ | $Abs Rel$ |
| - | - | - | - |
| Depth Estimation | [Download](https://huggingface.co/Sta8is/DINO-Foresight/blob/main/head_depth_pca1152.ckpt) | 89.1 | 0.108 |

or via command line:
```bash
wget https://huggingface.co/Sta8is/DINO-Foresight/resolve/main/head_depth_pca1152.ckpt
```

| Task | Checkpoint | $m$ | $11.25\degree$ | 
| - | - | - | - |
| Surface Normal Estimation | [Download](https://huggingface.co/Sta8is/DINO-Foresight/blob/main/head_normals_pca1152.ckpt) | 3.24 | 94.4 |

or via command line:
```bash
wget https://huggingface.co/Sta8is/DINO-Foresight/resolve/main/head_normals_pca1152.ckpt
```

## Inference 
In order to perform inference on the Cityscapes dataset use the training scripts and add the following arguments:
```bash
--ckpt_path "path/to/checkpoint" --eval_ckpt_only
```
