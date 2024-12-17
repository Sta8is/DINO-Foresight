import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
import glob
import os.path as osp
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import argparse
import timm
import einops

class cityscapes_sequence_data(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, subset='train', img_size=(448, 896)):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        self.files = glob.glob(osp.join(self.root_dir, subset, '**',"*.png"))
        self.files.sort()
        print(f'Found {len(self.files)} files')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

class dinov2(nn.Module):
    def __init__(self, dlayers=[2,5,8,11]):
        super(dinov2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.get_intermediate_layers(x,self.dlayers, reshape=False)
            if len(self.dlayers) > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x

class eva2_clip(nn.Module):
    def __init__(self, dlayers=[2,5,8,11], img_size=(448,896)):
        super(eva2_clip, self).__init__()
        self.model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, img_size = (448,896)) # pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])}
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.forward_intermediates(x, indices=self.dlayers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            x = torch.cat(x,dim=-1)
        return x

class sam(nn.Module):
    def __init__(self, dlayers=[2,5,8,11],img_size=(448,896)):
        super(sam, self).__init__()
        self.model =  timm.create_model('timm/samvit_base_patch16.sa1b', pretrained=True,  pretrained_cfg_overlay={'input_size': (3,img_size[0],img_size[1]),})
        self.model = self.model.eval()
        self.model = self.model.to('cuda')
        self.dlayers = dlayers
    
    def forward(self, x):
        with torch.no_grad():
            x = self.model.forward_intermediates(x, indices=self.dlayers, norm=False, intermediates_only=True)
            x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            x = torch.cat(x,dim=-1)
        return x

# Create a DataLoader for the dataset
def dataloader(root_dir,img_size=(448,896), batch_size=4, shuffle=False, num_workers=4,subset='train'):
    dataset = cityscapes_sequence_data(root_dir=root_dir, subset=subset, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

def parse_list(s, ): 
    return list(map(int, s.split(',')))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_extractor', type=str, default='dinov2', choices=['dinov2', 'eva2-clip', 'sam'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_components', type=int, default=1152)
    parser.add_argument('--dlayers', type=parse_list, default=[2,5,8,11])
    parser.add_argument('--img_size', type=parse_list, default=[448,896])
    parser.add_argument('--cityscapes_root', type=str, default='/storage/cityscapes/leftImg8bit')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    n_components = args.n_components
    dtype = torch.float32
    cs_loader = dataloader(root_dir=args.cityscapes_root, img_size=args.img_size, batch_size=args.batch_size, subset='train')
    n_batches = len(cs_loader)
    PCA = PCA(n_components=n_components)
    print(f'Number of batches: {n_batches}')
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if args.feature_extractor == 'dino':
        model  = dinov2(dlayers=args.dlayers).to(device).to(dtype)
    elif args.feature_extractor == 'eva2-clip':
        model = eva2_clip(dlayers=args.dlayers, img_size=args.img_size).to(device).to(dtype)
    elif args.feature_extractor == 'sam':
        model = sam(dlayers=args.dlayers, img_size=args.img_size).to(device).to(dtype)
    else:
        raise ValueError(f"feature extractor {args.feature_extractor} not supported")
    f_list = []
    for batch in tqdm(cs_loader):
        batch = batch.to(device).to(dtype)
        x = model(batch)
        f_list.append(x.flatten(end_dim=-2).float().cpu().numpy())

    f = np.concatenate(f_list)
    # Standardize the data
    mean = np.mean(f, axis=0)
    std = np.std(f, axis=0)
    f = (f - mean)/std
    print(f.shape)
    print('Fitting PCA')
    PCA.fit(f)
    print('PCA fitted')
    # Save the PCA model and mean/std in the same file
    checkpoint = {
        'pca_model': PCA,
        'mean': mean,
        'std': std
    }
    if len(args.dlayers) > 1:
        torch.save(checkpoint, args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers).replace(" ", "_")+'_'+str(n_components)+'.pth')
    else:
        torch.save(checkpoint, args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers[0])+'_'+str(n_components)+'.pth')

    # np.save('pca_mean_448_768.npy', mean)
    # np.save('pca_std_448_768.npy', std)
    # torch.save(PCA, 'pca_model_448_ms_768.pth')
    # torch.save(PCA, 'pca_model_224.pth')
    
    # Test
    print('Testing PCA')
    # Load the PCA model and mean/std
    # PCA = torch.load('pca_model_448_ms_768.pth')
    # mean = np.load('pca_mean_448_768.npy')
    # std = np.load('pca_std_448_768.npy')
    if len(args.dlayers) > 1:
        checkpoint = torch.load(args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers).replace(" ", "_")+'_'+str(n_components)+'.pth')
    else:
        checkpoint = torch.load(args.feature_extractor+'_pca_'+str(args.img_size[0])+'_l'+str(args.dlayers[0])+'_'+str(n_components)+'.pth')
    PCA = checkpoint['pca_model']
    mean = checkpoint['mean']
    std = checkpoint['std']
    cs_val_loader = dataloader(root_dir='/storage/cityscapes/leftImg8bit', batch_size=args.batch_size, subset='val')
    f_list = []
    for batch in tqdm(cs_val_loader):
        batch = batch.to(device).to(dtype)
        x = model(batch)
        f_list.append(x.flatten(end_dim=-2).float().cpu().numpy())
    
    f = np.concatenate(f_list)
    print(f.shape)
    print('Standardizing')
    f = (f - mean)/std
    print('Transforming')
    f_pca = PCA.transform(f)
    print(f_pca.shape)
    var = PCA.explained_variance_ratio_
    print(f'Explained variance: {var.sum()}')
    print(np.sum(np.var(f_pca, axis=0))/np.sum(np.var(f, axis=0)))
    