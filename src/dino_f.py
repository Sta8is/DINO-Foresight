import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import einops
from time import time
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.aggregation import MeanMetric
from src.attention_masked import MaskTransformer
import math
from dpt import DPTHead
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


class Dino_f(pl.LightningModule):
    def __init__(self,args):
        super(Dino_f,self).__init__()
        self.args = args
        self.sequence_length = args.sequence_length # 4
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.loss_type = args.loss_type
        self.img_size  = args.img_size
        self.d_layers = args.d_layers
        self.patch_size = 14 if self.args.feature_extractor in ['dino', 'eva2-clip'] else 16
        self.d_num_layers = len(self.d_layers) if isinstance(self.d_layers, list) else self.d_layers
        if not self.args.crop_feats and not self.args.sliding_window_inference:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size), self.img_size[1]//(self.patch_size))
        else:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size*2), self.img_size[1]//(self.patch_size*2))
        if self.args.feature_extractor == 'dino':
            self.dino_v2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_'+args.dinov2_variant, pretrained=True)
            for param in self.dino_v2.parameters():
                param.requires_grad = False
            self.dino_v2.eval()
        elif self.args.feature_extractor == 'eva2-clip':
            self.eva2clip = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, img_size = (self.img_size[0], self.img_size[1])) # pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])}
            for param in self.eva2clip.parameters():
                param.requires_grad = False
            self.eva2clip.eval()
        elif self.args.feature_extractor == 'sam':
            self.sam = timm.create_model('timm/samvit_base_patch16.sa1b', pretrained=True,  pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])})
            for param in self.sam.parameters():
                param.requires_grad = False
                self.sam.eval()
        if self.args.feature_extractor == 'dino':
            self.feature_dim = self.dino_v2.embed_dim
        elif self.args.feature_extractor == 'eva2-clip':
            self.feature_dim = self.eva2clip.embed_dim
        elif self.args.feature_extractor == 'sam':
            self.feature_dim = self.sam.embed_dim
        self.embedding_dim = self.d_num_layers * self.feature_dim
        if self.args.pca_ckpt:
            self.pca_dict = torch.load(self.args.pca_ckpt, weights_only=False)
            self.pca = self.pca_dict['pca_model']
            self.pca_mean = torch.nn.Parameter(torch.tensor(self.pca.mean_), requires_grad=False)
            self.pca_components = torch.nn.Parameter(torch.tensor(self.pca.components_), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.tensor(self.pca_dict['mean']), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(self.pca_dict['std']),requires_grad=False)
            self.embedding_dim = self.pca_components.shape[0]
        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
                                       seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size, use_first_last=args.use_first_last)
        self.train_mask_frames = args.train_mask_frames
        self.train_mask_mode = args.train_mask_mode
        self.masking = args.masking
        assert self.masking in ("half_half", "simple_replace", "half_half_previous")
        if self.masking in ("half_half", "half_half_previous"): # default 
            self.mask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.unmask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            if self.masking=="half_half":
                self.replace_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim//2)
        elif self.masking == "simple_replace":
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
            self.replace_vector = nn.Parameter(torch.zeros(1, 1, 1, 1, self.hidden_dim))
            torch.nn.init.normal_(self.replace_vector, std=.02)
            self.maskvit.fc_in = nn.Identity()
        self.activation = nn.Sigmoid() if args.output_activation == "sigmoid" else nn.Identity()
        # Necessary for evaluation
        self.mean_metric = MeanMetric()
        if self.args.eval_modality in ["segm", "depth", "surface_normals"]:   
            self.ignore_index = 255 if self.args.eval_modality == "segm" else 0
            self.head = DPTHead(nclass=self.args.num_classes,in_channels=self.feature_dim, features=self.args.nfeats,
                                use_bn=self.args.use_bn, out_channels=self.args.dpt_out_channels, use_clstoken=self.args.use_cls)
            self.patch_h = self.img_size[0] // self.patch_size
            self.patch_w = self.img_size[1] // self.patch_size
            # Load the head checkpoint
            if self.args.head_ckpt is not None:
                state_dict = {}
                for k, v in torch.load(self.args.head_ckpt)["state_dict"].items():
                    state_dict[k.replace("head.","")] = v
                self.head.load_state_dict(state_dict)
                self.head.eval()
                for param in self.head.parameters():
                    param.requires_grad = False
            if self.args.eval_modality == "segm":
                self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.args.num_classes, ignore_index=self.ignore_index, average=None)
            elif self.args.eval_modality == "depth":
                self.ignore_index = 0
                self.d1 = MeanMetric()
                self.d2 = MeanMetric()
                self.d3 = MeanMetric()
                self.abs_rel = MeanMetric()
                self.rmse = MeanMetric()
                self.log_10 = MeanMetric()
                self.rmse_log = MeanMetric()
                self.silog = MeanMetric()
                self.sq_rel = MeanMetric()
            elif self.args.eval_modality == "surface_normals":
                self.mean_ae = MeanMetric()
                self.median_ae = MeanMetric()
                self.rmse = MeanMetric()
                self.a1 = MeanMetric()
                self.a2 = MeanMetric()
                self.a3 = MeanMetric()
                self.a4 = MeanMetric()
                self.a5 = MeanMetric()
        self.batch_crops = []
        self.random_crop = T.RandomCrop(16,32)
        self.save_hyperparameters()
       

    def on_load_checkpoint(self, checkpoint):
        if self.args.high_res_adapt:
            # Interpolate positional embeddings
            pos_emb_h = checkpoint['state_dict']['maskvit.pos_embd.emb.d_1']
            pos_emb_h = F.interpolate(pos_emb_h.unsqueeze(0).unsqueeze(0), size=(self.img_size[0]//(self.patch_size), pos_emb_h.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_1'] = pos_emb_h.squeeze(0).squeeze(0)
            pos_emb_w = checkpoint['state_dict']['maskvit.pos_embd.emb.d_2']
            pos_emb_w = F.interpolate(pos_emb_w.unsqueeze(0).unsqueeze(0), size=(self.img_size[1]//(self.patch_size), pos_emb_w.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_2'] = pos_emb_w.squeeze(0).squeeze(0)
  
    def sliding_window(self, im, window_size, stride):
        B, SL, H, W, C = im.shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        windows = []
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                windows.append(im[:,:,i:i+ws_h,j:j+ws_w])
        return torch.stack(windows)

    def merge_windows(self, windows_res, original_shape, window_size, stride):
        B, SL, H, W, C = original_shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        merged = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        count = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        idx = 0
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                merged[:,:,i:i+ws_h,j:j+ws_w] += windows_res[idx].to(merged.device)
                count[:,:,i:i+ws_h,j:j+ws_w] += 1
                idx += 1
        merged /= count
        return merged


    def crop_feats(self, x, use_crop_params=False):
        B, SL, H, W, c= x.shape
        x = x.permute(0, 1, 4, 2, 3)
        if not use_crop_params:
            self.batch_crops = [self.random_crop.get_params(torch.zeros(H, W),(16,32)) for _ in range(B)]
        cropped_tensor = torch.stack([torch.stack([TF.crop(x[b, s], *self.batch_crops[b]) for s in range(SL)]) for b in range(B)])
        cropped_tensor = cropped_tensor.permute(0, 1, 3, 4, 2)
        return cropped_tensor

    def extract_features(self, x, reshape=False):
        with torch.no_grad():
            if self.args.feature_extractor == 'dino':
                x = self.dino_v2.get_intermediate_layers(x,n=self.d_layers, reshape=reshape)
            elif self.args.feature_extractor == 'eva2-clip':
                x = self.eva2clip.forward_intermediates(x, indices=self.d_layers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            elif self.args.feature_extractor == 'sam':
                x = self.sam.forward_intermediates(x, indices=self.d_layers, norm=False, intermediates_only=True) # Norm is False to avoide neck layer that reduces feature_dim to 256. Also output is in NCHW format
                x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            if self.d_num_layers > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x
            
    def get_mask_tokens(self, x, mode="arccos", mask_frames=1):
        B, sl, h, w, c = x.shape # x.shape [B,T,H,W,C]
        assert mask_frames <= sl
        if mode == "full_mask":
            if self.sequence_length == 7:
                assert mask_frames <= 3
            else:
                assert mask_frames == 1
            mask = torch.ones(B,mask_frames, h,w, dtype=torch.bool)
        else:
            r = torch.rand(B) # Batch size
            if mode == "linear":                # linear scheduler
                val_to_mask = r
            elif mode == "square":              # square scheduler
                val_to_mask = (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            # Create a mask of size [Batch,1,Height, Width] for the last frame of each sequence
            mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)

        # Create the mask for all frames, by concatenating the mask for the last frame to a tensor of zeros(no mask) for first T-1 frames
        mask = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask], dim=1).to(x.device)

        if self.masking in ("half_half", "half_half_previous"):
            # Create the mask_tokens tensor
            mask_tokens = mask.unsqueeze(-1).float()*self.mask_vector.expand(B,sl,h,w,-1) + (~mask.unsqueeze(-1)).float()*self.unmask_vector.expand(B,sl,h,w,-1)
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            if self.masking == "half_half_previous":
                # Replace the embedded tokens at masked locations with the embedded tokens from the previous frames
                replace = torch.cat((torch.zeros((B,1,h,w,embedded_tokens.shape[-1]), dtype=embedded_tokens.dtype, device=embedded_tokens.device), embedded_tokens[:,:-1]), dim=1)
            else:
                # Replace the embedded tokens at masked locations with the replace vector
                replace = self.replace_vector.expand(B,sl,h,w,-1)
            embedded_tokens = torch.where(mask.unsqueeze(-1), replace, embedded_tokens)
            # Concatenate the masked tokens to the embedded tokens. Only take half of each to get the right hidden size
            final_tokens = torch.cat((embedded_tokens,mask_tokens), dim=-1)
        elif self.masking=="simple_replace":
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            # Replace the embedded tokens at masked locations with the replace vector
            final_tokens = torch.where(mask.unsqueeze(-1), self.replace_vector.expand(B,sl,h,w,-1), embedded_tokens)

        return final_tokens, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
        :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
        :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step+1)[1:]
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return
        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.shape[1]*self.shape[2])
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.shape[1] * self.shape[2]) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def downsample_feats(self, x):
        # Accepts shape [B*T,H,W,C]
        BT,H,W,C = x.shape # shape [8*5, 16, 32, 3072]
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C,H,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_down = [self.downsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_down, dim=1), 1, -1) # [B*T,H//2,W//2,C]

    def upsample_feats(self, x):
        # Accepts shape [B*T,H//2,W//2,C]
        BT,H,W,C = x.shape
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C//2,H//2,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_up = [self.upsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_up, dim=1), 1, -1) # [B*T,H,W,C]

    def pca_transform(self, x):
        BT, HW, C = x.shape
        x = (x - self.mean) / self.std
        x = x - self.pca_mean
        x_pca = torch.matmul(x, self.pca_components.T)
        return x_pca

    def pca_inverse_transform(self, x):
        B, T, H, W, C = x.shape
        x = torch.matmul(x, self.pca_components) + self.pca_mean
        x = x * self.std + self.mean
        return x


    def preprocess(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W]. 
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1) # x.shape [B*T,C,H,W]
        x = self.extract_features(x) # [B*T,H*W,C]
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
        x = einops.rearrange(x, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        x = x.unflatten(dim=0, sizes=(B, self.sequence_length)) # [B,T,H,W,C]
        return x

    def postprocess(self, x):
        if self.args.pca_ckpt:
            x = self.pca_inverse_transform(x)
        return x
    
    def calculate_loss(self, x_pred, x_target):
        if self.args.loss_type == "MSE":
            loss = F.mse_loss(x_pred, x_target)
        elif self.args.loss_type == "SmoothL1":
            loss = F.smooth_l1_loss(x_pred, x_target, beta=self.args.beta_smoothl1)
        elif self.args.loss_type == "L1":
            loss = F.l1_loss(x_pred, x_target)
        return loss

    def forward_loss(self, x_pred, x_target, mask):
        B, T, H, W, C = x_pred.shape
        x_target = x_target[mask] 
        x_pred = x_pred[mask] 
        return self.calculate_loss(x_pred, x_target)
        
    def sample(self, x, sched_mode="arccos", step=15, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x = self.preprocess(x)
            B, SL, H, W, C = x.shape
            if not self.args.sliding_window_inference:
                if self.args.crop_feats:
                    x = self.crop_feats(x)
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens = self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                prediction = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x.shape[0]):
                    win = x[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens = self.forward(win, masked_soft_tokens, mask)
                    else:
                        # Instantiate scheduler
                        if isinstance(sched_mode, str):  # Standard ones
                            scheduler = self.adap_sche(step, mode=sched_mode)
                        else:  # Custom one
                            scheduler = sched_mode
                        final_tokens, loss = self.oracle_sample(x, masked_soft_tokens, mask, scheduler, step, mask_frames)
                    prediction = self.postprocess(final_tokens)
                    wins.append(prediction)
                prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, 3072), window_size, stride).to(x.device)
            return prediction, loss
            # return prediction, loss, final_tokens

    def sample_unroll(self, x, gt_feats, sched_mode="arccos", step=15, mask_frames=1, unroll_steps=3, ):
        self.maskvit.eval()
        with torch.no_grad():
            x = self.preprocess(x)
        B, SL, H, W, C = x.shape
        for i in range(unroll_steps):
            if not self.args.sliding_window_inference:
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens= self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                # x = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x_s = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x_s.shape[0]):
                    win = x_s[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens_win = self.forward(win, masked_soft_tokens, mask)
                    wins.append(final_tokens_win)
                final_tokens = self.merge_windows(torch.stack(wins), (B, SL, H, W, 1152), window_size, stride).to(x.device)
            x[:,-1] = final_tokens[:,-1]
            x = torch.cat((x[:,1:], x[:,-1].unsqueeze(1)), dim=1) # Mayve also try torch.zeros instead of x[:,-1]
        prediction = self.postprocess(x)
        loss = self.calculate_loss(prediction[:,-1].flatten(end_dim=-2), gt_feats.flatten(end_dim=-2))
        # return prediction, loss, x
        return prediction, loss

    def forward(self, x, masked_x, mask=None):
        if self.args.vis_attn:
            x_pred, attn = self.maskvit(masked_x,return_attn=True)
        else:
            x_pred, = self.maskvit(masked_x)
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        x_pred = self.activation(x_pred)
        # if self.args.predict_residuals:
        #     x_prev = torch.cat([torch.zeros(B,1,H,W,C).to(x.device), x[:,:-1]], dim=1)
        #     x_pred = x_pred + x_prev
        loss = self.forward_loss(x_pred=x_pred, x_target=x, mask=mask)
        if self.args.vis_attn:
            return loss, x_pred, attn
        else:
            return loss, x_pred

    

    def training_step(self, x, batch_idx):
        B = x.shape[0]
        x = self.preprocess(x)
        # Mask the encoded tokens
        if self.args.crop_feats:
            x = self.crop_feats(x)
        if self.sequence_length == 7:
            train_mask_frames = torch.randint(1, 4, (1,)) if self.training else 3
        else:
            train_mask_frames = self.train_mask_frames if self.training else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=train_mask_frames)
        # masked_x, mask = self.get_mask_tokens(x, mode="full_mask", mask_frames=train_mask_frames) # masked_x.shape [B,T,H,W,C], mask.shape [B,T,H,W,1]
        if self.args.vis_attn:
            loss, _, _ = self.forward(x, masked_x, mask)
        else:
            loss, _ = self.forward(x, masked_x, mask)
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        mask_ratio = mask[:,-1].float().mean().item() * 100
        self.log("Train/mask_ratio", mask_ratio, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode:
            return self.evaluation_step(batch, batch_idx)
        B = batch[0].shape[0]
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, batch_size=B, logger=True, on_step=True)

    def baseline_evaluation_step(self, x, gt_feats):
        B = x.shape[0]
        with torch.inference_mode():
            x = self.preprocess(x)
            x = self.postprocess(x)
            loss = self.calculate_loss(x[:,-2].flatten(end_dim=-2), gt_feats.flatten(end_dim=-2))
            x[:,-1] = x[:,-2]
        return x, loss

    def evaluation_step(self, batch, batch_idx):
        B, sl, C, H, W = batch[0].shape
        if self.args.eval_modality is None:
            data_tensor, gt_img= batch
        elif self.args.eval_modality == "segm":
            data_tensor, gt_img, gt_segm = batch
        elif self.args.eval_modality == "depth":
            data_tensor, gt_img, gt_depth = batch
        elif self.args.eval_modality == "surface_normals":
            data_tensor, gt_img, gt_normals = batch
        gt_feats = self.extract_features(gt_img)
        gt_feats = einops.rearrange(gt_feats, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        if self.args.evaluate_baseline:
            samples, loss = self.baseline_evaluation_step(data_tensor,gt_feats=gt_feats)
        else:
            if self.args.eval_midterm:
                samples, loss = self.sample_unroll(data_tensor,gt_feats,sched_mode=self.train_mask_mode,step=self.args.step, unroll_steps=3)
            else:
                samples, loss = self.sample(data_tensor,sched_mode=self.train_mask_mode,step=self.args.step)
        # Evaluation
        pred_feats = samples[:,-1]
        gt_feats = gt_feats.unsqueeze(1)
        if self.args.crop_feats:
            gt_feats = self.crop_feats(gt_feats, use_crop_params=True)
        gt_feats = gt_feats.squeeze(1)
        self.mean_metric.update(loss)
        mean_loss = self.mean_metric.compute()
        self.log('val/mean_loss', mean_loss, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        print(f"Iteration {batch_idx}: Validation Loss: {mean_loss:.4f}")
        if self.args.eval_modality == "segm":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_segm = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_segm = F.interpolate(pred_segm, size=(1024,2048), mode='bicubic', align_corners=False)
            self.iou_metric.update(pred_segm, gt_segm.squeeze(1))
            IoU = self.iou_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = torch.mean(IoU[11:])
            self.log('val/mIoU', mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/MO_mIoU', MO_mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            print(f"Validation mIoU: {mIoU:.4f}, Validation MO_mIoU: {MO_mIoU:.4f}")
        elif self.args.eval_modality == "depth":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_depth = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_depth = F.interpolate(pred_depth, size=(1024,2048), mode='bicubic', align_corners=False)
            pred_depth = pred_depth.argmax(dim=1)
            update_depth_metrics(pred_depth, gt_depth.squeeze(1), self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
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
        elif self.args.eval_modality == "surface_normals":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_normals = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_normals = F.interpolate(pred_normals, size=(1024,2048), mode='bicubic', align_corners=False)
            update_normal_metrics(pred_normals, gt_normals, self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            self.log('val/mean_ae', mean_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/median_ae', median_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a1', a1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a2', a2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a3', a3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a4', a4, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log('val/a5', a5, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        self.log('val/loss', loss, prog_bar=True, batch_size=data_tensor.shape[0], sync_dist=True, on_step=False, on_epoch=True, logger=True)
        
        
    def on_validation_epoch_end(self):
        if self.args.eval_mode:
            mean_loss = self.mean_metric.compute()
            print(f"Validation Loss: {mean_loss:.4f}")
            self.log_dict({'val/mean_loss': mean_loss}, prog_bar=True, logger=True)
            self.mean_metric.reset()
            if self.args.eval_modality == "segm":
                IoU = self.iou_metric.compute()
                mIoU = torch.mean(IoU)
                MO_mIoU = torch.mean(IoU[11:])
                print("mIoU = %10f" % (mIoU*100))
                print("MO_mIoU = %10f" % (MO_mIoU*100))
                self.log_dict({"val/mIoU": mIoU * 100, "val/MO_mIoU": MO_mIoU * 100}, logger=True, prog_bar=True)
                self.iou_metric.reset()
            elif self.args.eval_modality == "depth":
                d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
                print("d1 =%10f" % (d1), "d2 =%10f" % (d2), "d3 =%10f" % (d3), "abs_rel =%10f" % (abs_rel), "rmse =%10f" % (rmse), "log_10 =%10f" % (log_10), "rmse_log =%10f" % (rmse_log), "silog =%10f" % (silog), "sq_rel =%10f" % (sq_rel))
                self.log_dict({"d1":d1, "d2":d2, "d3":d3, "abs_rel":abs_rel, "rmse":rmse, "log_10":log_10, "rmse_log":rmse_log, "silog":silog, "sq_rel":sq_rel}, logger=True, prog_bar=True)
                reset_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            elif self.args.eval_modality == "surface_normals":
                mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
                print("mean_ae =%10f" % (mean_ae), "median_ae =%10f" % (median_ae), "rmse =%10f" % (rmse), "a1 =%10f" % (a1), "a2 =%10f" % (a2), "a3 =%10f" % (a3), "a4 =%10f" % (a4), "a5 =%10f" % (a5))
                self.log_dict({"mean_ae":mean_ae, "median_ae":median_ae, "rmse":rmse, "a1":a1, "a2":a2, "a3":a3, "a4":a4, "a5":a5}, logger=True, prog_bar=True)
                reset_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]