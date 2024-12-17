from torch.utils.data import Dataset, DataLoader
from PIL import Image
import transforms as T
import pytorch_lightning as pl
import torch
import os
import torchvision.transforms.v2 as Tv2
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None, modality='semseg', num_classes=19):
        self.root = root
        self.split = split
        self.transform = transform
        self.modality = modality
        self.num_classes = num_classes
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        if self.modality == 'segm':
            self.labels_dir = os.path.join(root, 'gtFine', split)
            self.replace_name = 'gtFine_labelTrainIds'
        elif self.modality == 'depth':
            self.labels_dir = os.path.join(root, 'leftImg8bit_sequence_depthv2', split)
            self.replace_name = 'leftImg8bit_depth'
        elif self.modality == 'surface_normals':
            self.labels_dir = os.path.join(root, 'leftImg8bit_normals', split)
            self.replace_name = 'leftImg8bit'
        else:
            raise ValueError(f"task {modality} not supported")
        self.image_files = []
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            label_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                self.image_files.append((os.path.join(img_dir, file_name),
                                         os.path.join(label_dir, file_name.replace('leftImg8bit', self.replace_name))))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path, label_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.modality in ['segm', 'depth']:
            label = Image.open(label_path)
        elif self.modality == "surface_normals":
            label_path = label_path.replace('png', 'npy')
            label = np.load(label_path)
            label = torch.from_numpy(label).permute(2, 0, 1)
        # Apply transformations
        image, label = self.transform(image, label)
        if self.modality == 'depth' and self.num_classes==1:
            return image, label/(self.num_classes-1)
        else:
            return image, label


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.img_size =  args.img_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_workers_val = args.num_workers_val
        self.modality = args.modality
        self.num_classes = args.num_classes
        if args.feature_extractor in ['dino', 'sam']:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif args.feature_extractor == 'eva2-clip':
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std = [0.26862954, 0.26130258, 0.27577711]
        else:
            raise ValueError(f"feature extractor {args.feature_extractor} not supported")
        self.transform_train = []
        self.transform_train += [T.Resize(self.img_size)]
        # self.transform_train += [T.RandomResize(min_size=int(0.5 * 512), max_size=int(2.0 * 512))]
        self.transform_train += [T.RandomHorizontalFlip(0.5)]
        self.transform_train += [T.RandomPhotometricDistort(0.5)]
        # self.transform_train += [T.RandomCrop((448,896))]
        if self.modality in ['segm', 'depth']:
            self.transform_train += [T.PILToTensor()]
        elif self.modality == 'surface_normals':
            self.transform_train += [T.PILToTensor(target_numpy=True)]
        self.transform_train += [T.ToDtype(torch.float, scale=True)]
        self.transform_train += [T.Normalize(mean=self.mean, std=self.std)]
        self.transform_train = T.Compose(self.transform_train)

        self.transform_val = []
        self.transform_val += [T.Resize(self.img_size)]
        if self.modality in ['segm', 'depth']:
            self.transform_val += [T.PILToTensor()]
        elif self.modality == 'surface_normals':
            self.transform_val += [T.PILToTensor(target_numpy=True)]
        self.transform_val += [T.ToDtype(torch.float, scale=True)]
        self.transform_val += [T.Normalize(mean=self.mean, std=self.std)]
        self.transform_val = T.Compose(self.transform_val)

    def train_dataloader(self):
        dataset = CityscapesDataset(self.data_path, split='train', transform=self.transform_train, 
                                    modality=self.modality, num_classes=self.num_classes)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        dataset = CityscapesDataset(self.data_path, split='val', transform=self.transform_val, 
                                    modality=self.modality, num_classes=self.num_classes)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers_val, shuffle=False, pin_memory=True)