from src.data import CS_VideoData 
from src.dino_f import Dino_f
import pytorch_lightning as pl
import torch 
import argparse
import os
import yaml
from pytorch_lightning.strategies import DDPStrategy
import numpy as np


def parse_tuple(x):
    return tuple(map(int, x.split(',')))

def parse_list(x):
    return list(map(int, x.split(',')))

parser = argparse.ArgumentParser()
# Data Parameters
parser.add_argument('--data_path', type=str, default='/home/ubuntu/cityscapes')
parser.add_argument('--dst_path', type=str, default=None)
parser.add_argument('--img_size', type=parse_tuple, default=(224,448))
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_workers_val', type=int, default=None, 
        help='(Optional) number of workers for the validation set dataloader. If None (default) it is the same as num_workers.')
parser.add_argument('--sequence_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_cond_frames', type=int, default=0)
parser.add_argument('--random_crop', action='store_true', default=False)
parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
parser.add_argument('--random_time_flip', action='store_true', default=False)
parser.add_argument('--timestep_augm', type=list, default=None, help='Probabilities for each timestep to be selected for augmentation starting from timestep 2 to \length of prob list e.g. [0.1,0.6,0.1,0.1,0.1] for timesteps [2,3,4,5,6]. If None, timestep [2,3,4] are selected with equal probability')
parser.add_argument('--no_timestep_augm', action='store_true', help='If True, no timestep augmentation is used (i.e., the num_frames_skip is always equal to 2 during training.)')
parser.add_argument('--use_fc_bias', action='store_true', help='Use bias for the fc_in and fc_out layers.')
parser.add_argument('--eval_modality', type=str, default=None, choices=[None, 'segm', 'depth', 'surface_normals'], help='Modality to be used for evaluation. If None, the input modality is used.')
# Trasformer Parameters
parser.add_argument('--feature_extractor', type=str, default='dino', choices=['dino', 'eva2-clip', 'sam'])
parser.add_argument('--dinov2_variant', type=str, default='vitb14_reg', choices=['vits14_reg','vitb14_reg'])
parser.add_argument('--d_layers', type=parse_list, default=[2,5,8,11])
parser.add_argument('--hidden_dim', type=int, default=768)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--layers', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--loss_type', type=str, default='SmoothL1',choices=['SmoothL1', 'MSE','L1'])
parser.add_argument('--beta_smoothl1', type=float, default=0.1)
parser.add_argument('--attn_dropout', type=float, default=0.3)
parser.add_argument('--step', type=int, default=2)
parser.add_argument('--masking', type=str, default='half_half', choices=['half_half', 'simple_replace', 'half_half_previous'])
parser.add_argument('--train_mask_mode', type=str, default='arccos', choices=['full_mask', 'arccos', 'linear', 'cosine', 'square'])
parser.add_argument('--seperable_attention', action='store_true', default=False)
parser.add_argument('--seperable_window_size', type=int, default=1)
parser.add_argument('--train_mask_frames', type=int, default=1)
parser.add_argument('--output_activation', type=str, default='none', choices=['none', 'sigmoid'])
parser.add_argument('--use_first_last', action='store_true', default=False)
parser.add_argument('--down_up_sample', action='store_true', default=False)
parser.add_argument('--pca_ckpt', type=str, default=None)
parser.add_argument('--crop_feats', action='store_true', default=False)
parser.add_argument('--sliding_window_inference', action='store_true', default=False)
parser.add_argument('--high_res_adapt', action='store_true', default=False, help='If True, the input images are resized to 448x896 instead of 224x448')
# DPT Head Parameters
parser.add_argument('--num_classes', type=int, default=19, choices=[19, 256, 3], help="19 Classes for segmentation, 256(classification) Depth or 3(regression) for surface normals")
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--use_cls', action='store_true', default=False)
parser.add_argument('--nfeats', type=int, default=256)
parser.add_argument('--dpt_out_channels', type=parse_list, default=[128, 256, 512, 512])
parser.add_argument('--head_ckpt', type=str, default=None)
# training parameters
parser.add_argument('--max_epochs', type=int, default=800) # 50220//372 == 135 epochs
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--single_step_sample_train', action='store_true', default=False)
parser.add_argument('--precision', type=str, default='32-true',choices=['16-true','16-mixed','32-true', '32'])
parser.add_argument('--ckpt', type=str, default=None, help='Path of a checkpoint to resume training')
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--accum_iter', type=int, default=1)
parser.add_argument('--warmup_p', type=float, default=0.0)
parser.add_argument('--lr_base', type=float, default=1e-3)
parser.add_argument('--gclip', type=float, default=1.0)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--vis_attn', action='store_true', default=False)
parser.add_argument('--eval_last', action='store_true', default=False)
parser.add_argument('--eval_ckpt_only', action='store_true', default=False)
parser.add_argument('--eval_mode_during_training', action='store_true', help='if activated (True) it uses the evaluation mode (i.e., step-by-step prediction and mIoU computation) during the training loop')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--use_val_to_train', action='store_true', default=False)
parser.add_argument('--use_train_to_val', action='store_true', default=False)
parser.add_argument('--evaluate_baseline', action='store_true', default=False)
parser.add_argument('--eval_midterm', action='store_true', default=False)

args = parser.parse_args()

args.eval_mode = args.eval_mode_during_training
pl.seed_everything(args.seed, workers=True)

data = CS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)


if args.precision == '32':
    args.precision = 32

if args.num_gpus > 1:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        assert gpus_per_node == torch.cuda.device_count()
        args.node = args.rank // gpus_per_node
    else:
        args.rank = 0
        args.world_size = args.num_gpus
        args.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device(args.gpu)
else:
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    args.node = 0
    args.device = torch.device('cuda:'+str(args.gpu))

print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
args.max_steps = (args.max_epochs * (len(data.train_dataloader()) // (args.num_gpus * args.accum_iter)))
args.warmup_steps = int(args.warmup_p * args.max_steps)
args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter
args.lr = (args.lr_base * args.effective_batch_size) / 8 # args.lr_base is specified for an effective batch-size of 8
print(f'Effective batch size:{args.effective_batch_size} lr_base={args.lr_base} lr={args.lr} max_epochs={args.max_epochs} - max_steps={args.max_steps}')

if not args.high_res_adapt:
    Dino_foresight = Dino_f(args)
else:
    Dino_foresight = Dino_f.load_from_checkpoint(args.ckpt,args=args,strict=False, map_location="cpu")

callbacks = []
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1, save_last=True)
callbacks.append(checkpoint_callback)
if args.dst_path is None:
    args.dst_path = os.getcwd()
if args.max_epochs < args.eval_freq:
    args.eval_freq = 1
trainer = pl.Trainer(
    accelerator='gpu',
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
   if args.ckpt and not args.high_res_adapt:
       trainer.fit(Dino_foresight,data,ckpt_path=args.ckpt)
   else:
       trainer.fit(Dino_foresight,data)
else:
    args.evaluate = True
    args.eval_last = True
    checkpoint_callback.last_model_path = args.ckpt
    data = CS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)

# Evaluation
if args.evaluate:
    args.eval_mode = True
    if not args.eval_last:
        print('Loading best model')
        checkpoint_path = checkpoint_callback.best_model_path
    else:
        print('Loading last model')
        checkpoint_path = checkpoint_callback.last_model_path

    print(f'checkpoint_path = {checkpoint_path}')
    Dino_foresight = Dino_f.load_from_checkpoint(checkpoint_path,args=args,strict=False, map_location="cpu")
    print('-----------Dino_foresight.eval_mode = ',Dino_foresight.args.eval_mode)
    Dino_foresight.to(args.device)
    Dino_foresight.eval()

    val_data_loader = data.val_dataloader()
    out_metrics = trainer.validate(model=Dino_foresight, dataloaders=val_data_loader)
    loss = out_metrics[0]['val/mean_loss']
    if args.eval_modality == "None":
        if args.rank==0:
            result_path = os.path.join(trainer.log_dir,'results.txt')
            with open(result_path,'w') as f:
                f.write(f'Mean Loss: {loss}\n')
    if args.eval_modality=='segm':
        mIoU = out_metrics[0]['val/mIoU']
        MO_mIoU = out_metrics[0]['val/MO_mIoU']
        if args.rank==0:
            result_path = os.path.join(trainer.log_dir,'results.txt')
            with open(result_path,'w') as f:
                f.write(f'Mean Loss: {loss}\n')
                f.write(f'mIoU: {mIoU}\n')
                f.write(f'MO_mIoU: {MO_mIoU}\n')
            print(f'Results saved in {result_path}')
    elif args.eval_modality == 'depth':
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
                f.write(f'Mean Loss: {loss}\n')
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
    elif args.eval_modality == 'surface_normals':
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
                f.write(f'Mean Loss: {loss}\n')
                f.write(f'mean_ae: {mean_ae}\n')
                f.write(f'median_ae: {median_ae}\n')
                f.write(f'rmse: {rmse}\n')
                f.write(f'a1: {a1}\n')
                f.write(f'a2: {a2}\n')
                f.write(f'a3: {a3}\n')
                f.write(f'a4: {a4}\n')
                f.write(f'a5: {a5}\n')
            print(f'Results saved at: {result_path}')
