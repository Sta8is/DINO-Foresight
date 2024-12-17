# Preparation of Labels for Library of Heads
For semantic/instance segmentation, we use the `leftImg8bit` and `gtFine`packages. We use [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) and [Lotus](https://github.com/EnVision-Research/Lotus) to generate depth and surface normals respectively. We provide the scripts to preprocess the Cityscapes dataset and generate the depth and surface normals.
Before proceeding, make sure you have the Cityscapes dataset downloaded and extracted.

## Semantic Segmentation
To train the semantic segmentation head `gtFine` needs to be processed using cityscapesScripts. Alternatively, you can download the processed dataset from [here](https://drive.google.com/file/d/1kd8KzEf8S5jlMAIPMoxOqW2cbjSlCs4w/view?usp=sharing). 
To download the preprocessed gtfine via command line:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1kd8KzEf8S5jlMAIPMoxOqW2cbjSlCs4w
```

## Depth
1. Clone the [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) repository, install the required packages and download the pre-trained [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) variant.
2. Place [run_depth_cs.py](run_depth_cs.py) in the DepthAnythingV2 directory.
3. Run the following command to generate the depth maps:
```bash
python run_depth_cs.py --encoder vitl --img-path /path/to/cityscapes/leftImg8bit --pred-only --grayscale
```

## Surface Normals
1. Clone the [Lotus](https://github.com/EnVision-Research/Lotus) repository, install the required packages and download the [normal regression model](https://huggingface.co/jingheya/lotus-normal-d-v1-0)
2. Place [infer_normals.sh](infer_normals.sh) in the Lotus directory and modify the base cityscapes path.
3. Run the following command to generate the surface normals:
```bash
bash infer_normals.sh
```
4. Move the generated surface normals to a new folder named `surface_normals` in the Cityscapes directory.

## Dataset Structure
The final dataset should be structured as follows:
```
cityscapes
│
├───leftImg8bit_sequence
│   ├───train
│   ├───val
│   ├───test
├───gtFine
│   ├───train
│   ├───val
│   ├───test
├───leftImg8bit
│   ├───train
│   ├───val
│   ├───test
├───leftImg8bit_depth
│   ├───train
│   ├───val
│   ├───test
├───leftImg8bit_normals
│   ├───train
│   ├───val
│   ├───test
```