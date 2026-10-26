import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from PIL import Image


def _normalization(feature_extractor):
    if feature_extractor in ['dino', 'sam']:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif feature_extractor == 'eva2-clip':
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    raise ValueError(f"feature extractor {feature_extractor} not supported")


class NuScenesRGBDataset(data.Dataset):
    """
    Dataset of front-camera (CAM_FRONT) video sequences of nuScenes.
    Every scene is one sequence made of ALL its camera frames (keyframes and sweeps, ~12 Hz), in temporal order.

    Args:
        data_path (str): The path to the nuScenes folder (the one that contains samples/, sweeps/ and v1.0-trainval/).
        args: The arguments (augmentation flags and eval_modality).
        sequence_length (int): The length of each sequence.
        img_size (tuple): The desired shape of the frames.
        subset (str): The subset of the dataset to use (train, val). Defaults to "train".

    Methods:
        __len__(): Returns the number of scenes in the dataset.
        __getitem__(idx): Retrieves a training window (or an evaluation window and its target frame) of scene idx.
    """

    def __init__(self, data_path, args, sequence_length, img_size, subset="train", eval_mode=False, eval_midterm=False, eval_modality=None, feature_extractor="dino"):
        super().__init__()
        # The devkit is only needed for nuScenes: pip install nuscenes-devkit
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
        assert args.eval_modality in (None, "depth", "surface_normals"), "nuScenes has no segmentation labels"
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.subset = subset
        self.eval_mode = eval_mode
        self.eval_midterm = eval_midterm
        self.eval_modality = args.eval_modality
        self.feature_extractor = feature_extractor
        self.camera_name = 'CAM_FRONT'
        # Label folders of the depth / surface normal modalities: siblings of the nuScenes folder
        root = data_path.rstrip('/')
        self.depth_path = root + '_depthv2'
        self.normals_path = root + '_normals'
        self.augmentations = {
                "random_crop" : args.random_crop,
                "random_horizontal_flip" : args.random_horizontal_flip,
                "random_time_flip" : args.random_time_flip,
                "timestep_augm" : args.timestep_augm,
                "no_timestep_augm" : args.no_timestep_augm}
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
        scene_names = create_splits_scenes()[subset]
        self.sequences = [self._scene_frames(scene) for scene in self.nusc.scene if scene['name'] in scene_names]

    def _scene_frames(self, scene):
        """File paths of all the front-camera frames of a scene, in temporal order."""
        cam_token = self.nusc.get('sample', scene['first_sample_token'])['data'][self.camera_name]
        paths = []
        while cam_token:
            cam_data = self.nusc.get('sample_data', cam_token)
            paths.append(osp.join(self.nusc.dataroot, cam_data['filename']))
            cam_token = cam_data['next']
        return paths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames_filepaths = self.sequences[idx]
        if self.eval_mode:
            return process_evalmode(frames_filepaths, self.img_size, self.sequence_length, self.eval_midterm, self.eval_modality,
                                    feature_extractor=self.feature_extractor, data_path=self.data_path,
                                    depth_path=self.depth_path, normals_path=self.normals_path)
        return process_trainmode(frames_filepaths, self.img_size, self.subset, self.augmentations, self.sequence_length, feature_extractor=self.feature_extractor)


def process_trainmode(frames_path, img_size, subset, augmentations, sequence_length=5, num_frames_skip=0, feature_extractor="dino"):
    if subset=="val":
        num_frames_skip = 2
        step = num_frames_skip + 1
        start_idx = 20 - step*sequence_length + num_frames_skip
    else:
        if augmentations["no_timestep_augm"] is True:
            num_frames_skip = 2
        elif augmentations["timestep_augm"] is not None:
            num_frames_skip = np.random.choice(list(range(1,len(augmentations["timestep_augm"])+1)),p=augmentations["timestep_augm"])
        else:
            num_frames_skip = np.random.randint(1,4) # [1,3] with equal probabilities 4 is excluded
        step = num_frames_skip + 1 # [2,4] with equal probabilities or [2,N] with non-equal probabilities
        start_idx = np.random.randint(0, len(frames_path) - step*sequence_length + num_frames_skip + 1)
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    # Load frames as tensors and apply transformations
    if augmentations["random_time_flip"] == True and subset=="train":
        sequence_frames_path = sequence_frames_path[::-1] if np.random.rand()>0.5 else sequence_frames_path
    sequence_frames = [Image.open(frame).convert('RGB') for frame in sequence_frames_path]
    W, H = sequence_frames[0].size # PIL IMAGE
    if augmentations["random_crop"] == True and subset=="train":
        s_f = np.random.rand()/2 + 0.5 # [0.5, 1]
        size = (int(H*s_f),int(W*s_f))
        i, j, h, w = T.RandomCrop(size).get_params(sequence_frames[0], output_size=size)
        sequence_frames = [TF.crop(frame,i,j,h,w) for frame in sequence_frames]
    if augmentations["random_horizontal_flip"] == True and subset=="train":
        sequence_frames = [TF.hflip(frame) for frame in sequence_frames] if np.random.rand()>0.5 else sequence_frames
    mean, std = _normalization(feature_extractor)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    return torch.stack([transform(frame) for frame in sequence_frames], dim=0)


def process_evalmode(frames_path, img_size, sequence_length=5, eval_midterm=False, eval_modality=None, feature_extractor="dino",
                     data_path=None, depth_path=None, normals_path=None):
    num_frames_skip = 2
    step = num_frames_skip + 1
    # The target is frame 28 of the scene. The context window ends 3 frames before it (short term),
    # or 2 rollouts of 3 frames earlier (mid term).
    start_idx = 29 - step*sequence_length + num_frames_skip
    if eval_midterm and sequence_length<7:
        start_idx -= 6
    sequence_frames_path = frames_path[start_idx : start_idx + step*sequence_length : step]
    mean, std = _normalization(feature_extractor)
    transform = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=mean, std=std)])
    sequence_tensor = torch.stack([transform(Image.open(frame).convert('RGB')) for frame in sequence_frames_path], dim=0)
    gt_path = frames_path[28]
    gt_img = transform(Image.open(gt_path).convert('RGB'))
    if eval_modality is None:
        return sequence_tensor, gt_img
    rel_path = osp.relpath(gt_path, data_path)   # e.g. samples/CAM_FRONT/<name>.jpg
    if eval_modality == "depth":
        gt_depth = T.PILToTensor()(Image.open(osp.join(depth_path, rel_path)))   # [1, H, W], 8-bit depth
        return sequence_tensor, gt_img, gt_depth
    elif eval_modality == "surface_normals":
        gt_normals = np.load(osp.join(normals_path, osp.splitext(rel_path)[0] + ".npy"))   # [H, W, 3]
        return sequence_tensor, gt_img, torch.from_numpy(gt_normals).permute(2, 0, 1)


class NS_VideoData(pl.LightningDataModule):
    """
    LightningDataModule for loading nuScenes front-camera video data.

    Args:
        arguments: An object containing the required arguments for data loading.
        subset (str): The subset of the data to load. Default is "train".
    """

    def __init__(self, arguments, subset="train", batch_size=8):
        super().__init__()
        self.data_path = arguments.data_path
        self.subset = subset  # ["train","val"]
        self.sequence_length = arguments.sequence_length
        self.batch_size = batch_size
        self.img_size = arguments.img_size
        self.arguments = arguments
        self.eval_midterm = arguments.eval_midterm
        self.eval_modality = arguments.eval_modality
        self.num_workers = arguments.num_workers
        self.num_workers_val = arguments.num_workers if arguments.num_workers_val is None else arguments.num_workers_val
        self.eval_mode = arguments.eval_mode
        self.use_val_to_train = arguments.use_val_to_train
        self.use_train_to_val = arguments.use_train_to_val
        self.feature_extractor = arguments.feature_extractor

    def _dataset(self, subset, eval_mode):
        return NuScenesRGBDataset(self.data_path, self.arguments, self.sequence_length, self.img_size, subset, eval_mode, self.eval_midterm, self.eval_modality, self.feature_extractor)

    def _dataloader(self, subset, shuffle=True, drop_last=False, eval_mode=False, num_workers=None):
        dataset = self._dataset(subset, eval_mode)
        return data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers if num_workers is None else num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last)

    def train_dataloader(self):
        train_subset = "val" if self.use_val_to_train else "train"
        return self._dataloader(subset=train_subset, drop_last=True, eval_mode=False)

    def val_dataloader(self):
        val_subset = "train" if self.use_train_to_val else "val"
        return self._dataloader(subset=val_subset, shuffle=False, eval_mode=self.eval_mode, num_workers=self.num_workers_val)

    def test_dataloader(self):
        return self._dataloader(subset="val", shuffle=False, eval_mode=self.eval_mode, num_workers=self.num_workers_val)
