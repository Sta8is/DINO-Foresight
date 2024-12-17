import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import sys, os
import einops
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from dpt import DPTHead
import numpy as np
import timm

def update_depth_metrics(pred, gt, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    valid_pixels = gt > 0
    pred = pred[valid_pixels]
    gt = gt[valid_pixels]
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    d1_m.update(d1)
    d2_m.update(d2)
    d3_m.update(d3)
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.float().mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    abs_rel_m.update(abs_rel)
    rmse_m.update(rmse)
    log_10_m.update(log_10)
    rmse_log_m.update(rmse_log)
    silog_m.update(silog)
    sq_rel_m.update(sq_rel)

    
def compute_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1 = d1_m.compute()
    d2 = d2_m.compute()
    d3 = d3_m.compute()
    abs_rel = abs_rel_m.compute()
    rmse = rmse_m.compute()
    log_10 = log_10_m.compute()
    rmse_log = rmse_log_m.compute()
    silog = silog_m.compute()
    sq_rel = sq_rel_m.compute()
    return d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
    

def reset_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1_m.reset()
    d2_m.reset()
    d3_m.reset()
    abs_rel_m.reset()
    rmse_m.reset()
    log_10_m.reset()
    rmse_log_m.reset()
    silog_m.reset()
    sq_rel_m.reset()

def update_normal_metrics(pred, gt, mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred, gt, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    mean_ae = pred_error.mean()
    median_ae = pred_error.median()
    rmse = torch.sqrt((pred_error ** 2).mean())
    a1 = 100*(pred_error < 5).float().mean()
    a2 = 100*(pred_error < 7.5).float().mean()
    a3 = 100*(pred_error < 11.25).float().mean()
    a4 = 100*(pred_error < 22.5).float().mean()
    a5 = 100*(pred_error < 30).float().mean()
    mean_ae_m.update(mean_ae)
    median_ae_m.update(median_ae)
    rmse_m.update(rmse)
    a1_m.update(a1)
    a2_m.update(a2)
    a3_m.update(a3)
    a4_m.update(a4)
    a5_m.update(a5)

def compute_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae = mean_ae_m.compute()
    median_ae = median_ae_m.compute()
    rmse = rmse_m.compute()
    a1 = a1_m.compute()
    a2 = a2_m.compute()
    a3 = a3_m.compute()
    a4 = a4_m.compute()
    a5 = a5_m.compute()
    return mean_ae, median_ae, rmse, a1, a2, a3, a4, a5

def reset_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae_m.reset()
    median_ae_m.reset()
    rmse_m.reset()
    a1_m.reset()
    a2_m.reset()
    a3_m.reset()
    a4_m.reset()
    a5_m.reset()


class DinoV2DPTModel(pl.LightningModule):
    def __init__(self, args):
        super(DinoV2DPTModel, self).__init__()
        self.args = args
        self.img_size = args.img_size
        if self.args.feature_extractor == 'dino':
            # Load DINO v2 backbone
            self.backbone =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        elif self.args.feature_extractor == 'eva2-clip':
            self.backbone = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, img_size = (self.img_size[0], self.img_size[1])) # pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])}
        elif self.args.feature_extractor == 'sam':
            self.backbone = timm.create_model('timm/samvit_base_patch16.sa1b', pretrained=True,  pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])})
        self.patch_size = 14 if self.args.feature_extractor in ['dino', 'eva2-clip'] else 16
        self.patch_h = self.img_size[0] // self.patch_size 
        self.patch_w = self.img_size[1] // self.patch_size
        self.emb_dim = self.backbone.embed_dim
        assert type(self.args.dlayers) == list or type(self.args.dlayers) == int, "dlayers should be a list or an integer"
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        if self.args.pca_ckpt:
            self.pca_dict = torch.load(self.args.pca_ckpt)
            self.pca = self.pca_dict['pca_model']
            self.pca_mean = torch.nn.Parameter(torch.tensor(self.pca.mean_), requires_grad=False)
            self.pca_components = torch.nn.Parameter(torch.tensor(self.pca.components_), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.tensor(self.pca_dict['mean']), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(self.pca_dict['std']),requires_grad=False)
        if self.args.modality == "segm":
            from torchmetrics.classification import JaccardIndex
            self.ignore_index = 255
            self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.args.num_classes, ignore_index=self.ignore_index, average=None)
        elif self.args.modality == "depth":
            self.ignore_index = 0
            from torchmetrics.aggregation import MeanMetric
            self.d1 = MeanMetric()
            self.d2 = MeanMetric()
            self.d3 = MeanMetric()
            self.abs_rel = MeanMetric()
            self.rmse = MeanMetric()
            self.log_10 = MeanMetric()
            self.rmse_log = MeanMetric()
            self.silog = MeanMetric()
            self.sq_rel = MeanMetric()
            if self.args.num_classes == 256:
                self.ignore_index = 0
            else:
                pass # Depth as regression problem
        elif self.args.modality == "surface_normals":
            from torchmetrics.aggregation import MeanMetric
            self.mean_ae = MeanMetric()
            self.median_ae = MeanMetric()
            self.rmse = MeanMetric()
            self.a1 = MeanMetric()
            self.a2 = MeanMetric()
            self.a3 = MeanMetric()
            self.a4 = MeanMetric()
            self.a5 = MeanMetric()
        self.head = DPTHead(nclass=self.args.num_classes, in_channels=self.emb_dim, features=self.args.nfeats, 
                            use_bn=self.args.use_bn, out_channels=self.args.dpt_out_channels, use_clstoken=self.args.use_cls)
        for param in self.head.scratch.refinenet4.resConfUnit1.parameters():
            param.requires_grad = False
        self.save_hyperparameters()
        # Set to False because we only care about the Head
        self.strict_loading = False
            
    def state_dict(self):
        # Save only the head weights. Backbone weights are not needed (Frozen)
        return {k: v for k, v in super().state_dict().items() if "head" in k}
    
    def pca_transform(self, x):
        BT, HW, C = x.shape
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        x = x - self.pca_mean
        x_pca = torch.matmul(x, self.pca_components.T)
        return x_pca
    
    def pca_inverse_transform(self, x):
        BT, HW, C = x.shape
        x = torch.matmul(x, self.pca_components) + self.pca_mean
        x = x * self.std.to(x.device) + self.mean.to(x.device)
        return x

    def forward(self, x):
        with torch.no_grad():
            if self.args.feature_extractor == 'dino':
                x = self.backbone.get_intermediate_layers(x, self.args.dlayers, return_class_token=self.args.use_cls, norm=True)
            elif self.args.feature_extractor == 'eva2-clip':
                x = self.backbone.forward_intermediates(x, indices=self.args.dlayers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            elif self.args.feature_extractor == 'sam':
                x = self.backbone.forward_intermediates(x, indices=self.args.dlayers, norm=False, intermediates_only=True) # Norm is False to avoide neck layer that reduces feature_dim to 256. Also output is in NCHW format
                x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            x = torch.cat(x, dim=2)
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
            x = self.pca_inverse_transform(x)
        x = [x[:,:,i*self.backbone.embed_dim:(i+1)*self.backbone.embed_dim] for i in range(len(self.args.dlayers))]
        out = self.head(x, self.patch_h, self.patch_w)
        out = F.interpolate(out, size=self.img_size, mode='bicubic', align_corners=False)
        return out

    def training_step(self, batch, batch_idx):
        B = batch[0].shape[0]
        x, gt = batch
        out = self.forward(x)
        if self.args.modality == "surface_normals" and self.args.num_classes == 3:
            l2_loss = torch.mean(torch.sum(torch.square(out - gt), dim=1)) # L2 loss
            cos_sim = F.cosine_similarity(out, gt, dim=1) 
            cos_loss = torch.mean(1 - cos_sim)
            loss = l2_loss + cos_loss
        else:
            ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss = ce_loss(out, gt)
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, gt = batch
        out = self.forward(x)
        if self.args.hflip_tta:
            out_hflip = self.forward(TF.hflip(data))
            out_hflip = TF.hflip(out_hflip)
            out = 0.5 * (out + out_hflip)
        if self.args.modality == "segm":
            self.iou_metric.update(out, gt)
            IoU = self.iou_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = torch.mean(IoU[11:])
            self.log('val/mIoU', mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/MO_mIoU', MO_mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        elif self.args.modality == "depth":
            out = out.argmax(dim=1)
            update_depth_metrics(out, gt, self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            self.log('val/d1', d1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d2', d2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/d3', d3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/abs_rel', abs_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/log_10', log_10, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse_log', rmse_log, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/silog', silog, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/sq_rel', sq_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        if self.args.modality == "surface_normals":
            update_normal_metrics(out, gt, self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            self.log('val/mean_ae', mean_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/median_ae', median_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a1', a1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a2', a2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a3', a3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a4', a4, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a5', a5, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
    
    def on_validation_epoch_end(self):
        if self.args.modality == "segm":
            IoU = self.iou_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = torch.mean(IoU[11:])
            print("mIoU = %10f" % (mIoU*100))
            print("MO_mIoU = %10f" % (MO_mIoU*100))
            self.log_dict({"mIoU": mIoU * 100, "MO_mIoU": MO_mIoU * 100}, logger=True, prog_bar=True)
            self.iou_metric.reset()
        elif self.args.modality == "depth":
            d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            print("d1 =%10f" % (d1), "d2 =%10f" % (d2), "d3 =%10f" % (d3), "abs_rel =%10f" % (abs_rel), "rmse =%10f" % (rmse), "log_10 =%10f" % (log_10), "rmse_log =%10f" % (rmse_log), "silog =%10f" % (silog), "sq_rel =%10f" % (sq_rel))
            self.log_dict({"d1":d1, "d2":d2, "d3":d3, "abs_rel":abs_rel, "rmse":rmse, "log_10":log_10, "rmse_log":rmse_log, "silog":silog, "sq_rel":sq_rel}, logger=True, prog_bar=True)
            reset_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
        elif self.args.modality == "surface_normals":
            mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            print("mean_ae =%10f" % (mean_ae), "median_ae =%10f" % (median_ae), "rmse =%10f" % (rmse), "a1 =%10f" % (a1), "a2 =%10f" % (a2), "a3 =%10f" % (a3), "a4 =%10f" % (a4), "a5 =%10f" % (a5))
            self.log_dict({"mean_ae":mean_ae, "median_ae":median_ae, "rmse":rmse, "a1":a1, "a2":a2, "a3":a3, "a4":a4, "a5":a5}, logger=True, prog_bar=True)
            reset_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
    
    def configure_optimizers(self):
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        elif self.args.optimizer == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay, betas=(0.9, 0.999))
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} not implemented")
        warmup_steps = self.args.warmup_steps
        if self.args.scheduler == "poly":
            main_lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.args.max_steps, power=1.0) 
        elif self.args.scheduler == "cosine":
            main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_steps)
        if warmup_steps > 0:
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps)
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        # For Channed scheduler is not neccessary to set total_iters = max_steps - warmup_steps
        scheduler = optim.lr_scheduler.ChainedScheduler([warmup_lr_scheduler, main_lr_scheduler])
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]