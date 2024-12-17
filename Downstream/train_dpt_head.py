import argparse
from pytorch_lightning.strategies import DDPStrategy
from data import CityscapesDataModule
from dinov2dpt import DinoV2DPTModel
import pytorch_lightning as pl
import torch
import os

parser = argparse.ArgumentParser()

def parse_list(s, ): 
    return list(map(int, s.split(',')))

def parse_list_str(s, ):
    return list(map(str, s.split(',')))

def parse_tuple(s):
    return tuple(map(int, s.split(',')))

# Data Parameters
parser.add_argument('--data_path', type=str, default="/home/ubuntu/cityscapes")
parser.add_argument('--img_size', type=parse_tuple, default=(448,896))
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_workers_val', type=int, default=8, help="(Optional) number of workers for the validation set dataloader. If None (default) it is the same as num_workers.")
parser.add_argument('--modality', type=str, default="segm", choices=["segm", "depth","surface_normals"])
# Model Parameters
parser.add_argument('--feature_extractor', type=str, default='dino', choices=['dino', 'eva2-clip','sam'])
parser.add_argument('--num_classes', type=int, default=19, choices=[19, 256, 1, 3], help="19 Classes for segmentation, 256(classification), 3 for surface normals(regression)")
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--dlayers', type=parse_list, default=[2,5,8,11], help="Dino layers to use (Starting from 0)")
parser.add_argument('--use_cls', action='store_true', default=False)
parser.add_argument('--nfeats', type=int, default=256)
parser.add_argument('--dpt_out_channels', type=parse_list, default=[128, 256, 1024, 1024]) #Old Default [256, 512, 1024, 1024]
parser.add_argument('--pca_ckpt', type=str, default=None, help="Path to the PCA checkpoint. If specified, the PCA matrices are loaded from this checkpoint and head is trained on reconstructed features.")
# Training Parameters
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--warmup_p', type=float, default=0.0)
parser.add_argument('--lr_base', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default="adamw", choices=["adam", "adamw"])
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--scheduler', type=str, default="poly", choices=["cosine", "poly"])
parser.add_argument("--precision", type=str, default="32-true",choices=["16-true","16-mixed","32-true"])
parser.add_argument("--ckpt", type=str, default=None, help="Path of a checkpoint to resume training or evaluate")
parser.add_argument("--dst_path", type=str, default=None, help="Path to save checkpoints and logs")
parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate every eval_freq epochs")
parser.add_argument("--gclip", type=float, default=0.0, help="Gradient clipping value")
parser.add_argument("--accum_iter", type=int, default=1, help="Number of iterations to accumulate gradients")
parser.add_argument("--eval_ckpt_only", action='store_true', default=False, help="Evaluate only the checkpoint without training")
parser.add_argument("--eval_last", action='store_true', default=False)
parser.add_argument('--hflip_tta', action='store_true', default=False, help="Horizontal flip test time augmentation")
args = parser.parse_args()

pl.seed_everything(args.seed, workers=True)

data = CityscapesDataModule(args)
if args.num_gpus > 1:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()
        args.node = args.rank // gpus_per_node
    else:
        args.rank = 0
        args.world_size = args.num_gpus
        args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device(args.gpu)
else:
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.node = 0
    args.device = torch.device("cuda:"+str(args.gpu))

print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
len_train = len(data.train_dataloader())
args.max_steps = (args.max_epochs * len_train // (args.num_gpus * args.accum_iter))
args.warmup_steps = int(args.warmup_p * args.max_steps)
args.effective_batch_size = args.batch_size * args.num_gpus * args.accum_iter
args.lr = (args.lr_base * args.effective_batch_size) / 8 # args.lr_base is specified for an effective batch-size of 8
print(f'Effective batch size:{args.effective_batch_size} lr_base={args.lr_base} lr={args.lr} max_epochs={args.max_epochs} - max_steps={args.max_steps} - warmup_steps={args.warmup_steps}')

callbacks = []
# monitor_val = "val/mIoU" if args.modality == "segm" else "val/abs_rel"
if args.modality == "segm":
    monitor_val = "val/mIoU"
    mode = "max"
elif args.modality == "depth":
    monitor_val = "val/abs_rel"
    mode = "min"
elif args.modality == "surface_normals":
    monitor_val = "val/mean_ae"
    mode = "min"
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor_val, mode=mode, save_top_k=1, save_last=True)
callbacks.append(checkpoint_callback)
if args.dst_path is None:
    args.dst_path = os.getcwd()
if args.max_epochs < args.eval_freq:
    args.eval_freq = 1
trainer = pl.Trainer(
    accelerator="gpu",
    strategy=(DDPStrategy(find_unused_parameters=False) if args.num_gpus > 1 else 'auto'),
    devices=args.num_gpus,
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    gradient_clip_val=args.gclip,
    default_root_dir=args.dst_path,
    precision=args.precision,
    log_every_n_steps=5,
    check_val_every_n_epoch=args.eval_freq,
    accumulate_grad_batches=args.accum_iter)

if not args.eval_ckpt_only:
    model = DinoV2DPTModel(args)
    trainer.fit(model, data)
else:
    checkpoint_callback.best_model_path = args.ckpt

if not args.eval_last:
    print("Loading best model")
    checkpoint_path = checkpoint_callback.best_model_path
else:
    print("Loading last model")
    checkpoint_path = checkpoint_callback.last_model_path

print(f'checkpoint_path = {checkpoint_path}')
model = DinoV2DPTModel.load_from_checkpoint(checkpoint_path, args=args,strict=False)
model.eval()
val_data_loader = data.val_dataloader()
out_metrics = trainer.validate(model=model, dataloaders=val_data_loader)
if args.modality == 'segm':
    mIoU = out_metrics[0]["mIoU"]
    MO_mIoU = out_metrics[0]["MO_mIoU"]
    if args.rank == 0:
        # Save mIoU and MO_mIoU to a text file
        result_path = os.path.join(trainer.log_dir, 'results.txt')
        with open(result_path, 'w') as f:
            f.write(f'mIoU: {mIoU}\n')
            f.write(f'MO_mIoU: {MO_mIoU}\n')
        print(f'Results saved at: {result_path}')
elif args.modality == 'depth':
    d1 = out_metrics[0]["d1"]
    d2 = out_metrics[0]["d2"]
    d3 = out_metrics[0]["d3"]
    abs_rel = out_metrics[0]["abs_rel"]
    rmse = out_metrics[0]["rmse"]
    rmse_log = out_metrics[0]["rmse_log"]
    silog = out_metrics[0]["silog"]
    sq_rel = out_metrics[0]["sq_rel"]
    log_10 = out_metrics[0]["log_10"]
    if args.rank == 0:
        # Save d1 to a text file
        result_path = os.path.join(trainer.log_dir, 'results.txt')
        with open(result_path, 'w') as f:
            f.write(f'd1: {d1}\n')
            f.write(f'd2: {d2}\n')
            f.write(f'd3: {d3}\n')
            f.write(f'abs_rel: {abs_rel}\n')
            f.write(f'rmse: {rmse}\n')
            f.write(f'rmse_log: {rmse_log}\n')
            f.write(f'sq_rel: {sq_rel}\n')
            f.write(f'log_10: {log_10}\n')
            f.write(f'silog: {silog}\n')
        print(f'Results saved at: {result_path}')
elif args.modality == 'surface_normals':
    mean_ae = out_metrics[0]["mean_ae"]
    median_ae = out_metrics[0]["median_ae"]
    rmse = out_metrics[0]["rmse"]
    a1 = out_metrics[0]["a1"]
    a2 = out_metrics[0]["a2"]
    a3 = out_metrics[0]["a3"]
    a4 = out_metrics[0]["a4"]
    a5 = out_metrics[0]["a5"]
    if args.rank == 0:
        # Save d1 to a text file
        result_path = os.path.join(trainer.log_dir, 'results.txt')
        with open(result_path, 'w') as f:
            f.write(f'mean_ae: {mean_ae}\n')
            f.write(f'median_ae: {median_ae}\n')
            f.write(f'rmse: {rmse}\n')
            f.write(f'a1: {a1}\n')
            f.write(f'a2: {a2}\n')
            f.write(f'a3: {a3}\n')
            f.write(f'a4: {a4}\n')
            f.write(f'a5: {a5}\n')
        print(f'Results saved at: {result_path}')
