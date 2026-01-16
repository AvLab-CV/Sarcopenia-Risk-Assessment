# Skeleton-Based Motion Pattern Recognition for Sarcopenia Risk Assessment

Yu-Hsuan Chiu, Aldo Acevedo Onieva, Gee-Sern Hsu, Jiunn-Horng Kang

## Installation

Requirements:

- Python >= 3.9
- CUDA >= 11.3

```sh
git clone https://github.com/Sinono3/sarcopenia
cd sarcopenia
uv sync
source .venv/bin/activate
```

## Usage

### Inference

You need to record videos of the subjects performing tandem gait, and save them into a directory. For example, let's say you have all videos on a directory `/home/user/sarcopenia/testvids`.

If the video is not square, you need to resize it to a square aspect-ratio to make it compatible with the pose estimation model. To facilitate this, we have provided the script **`video_resize.py`**. This script takes all videos from one directory, resizes them to 700x700 (you can adjust this), and saves them to another directory. The method for resizing adds black bars, i.e. it makes the picture fit to the resolution, _not_ fill. This is important, because we don't want to cut-off the subject. Example:

```sh
python video_resize.py /home/user/sarcopenia/testvids /home/user/sarcopenia/testvids_700x700
```

Once you have the resized videos, you need to estimate the poses of the subjects in the videos. For this, we have provided the script **`mp4_to_skeleton.py`**. This script reads video files from a directory and outputs a single `.npz` file containing all the skeletons.

```sh
python mp4_to_skeleton.py /home/user/sarcopenia/testvids_700x700 skeletons/testvids.npz
```

The stability classifier (SkateFormer model) expects the skeleton data in Kinect V2 format,
comprised by 25 joints. Our pose estimation method outputs in the 17-joint H36M format.
Because of this discrepancy, we must augment our 17-joint skeleton to 25 joints,
we provide a script for this purpose:

```sh
python ./skeleton_to_ntu_format.py --input-format h36m17 skeletons/testvids.npz skeletons/testvids_ntu.npz
```

Now, with the stability classifier trained, we can do the actual testing of the full pipeline, and stratify the subjects
into risk groups depending on their instability rate. The script `inference.py` does this. But you first must modify the configuration file `config/sarcopenia/inference.yaml`, specifically the lines 5, 6 and 11:

- `work_dir`: output directory of the inference. There will be a CSV containing the sliding window instability rates.
- `weights`: weights to the pretrained stability classifier `.pt` file. You can download this [here](https://drive.google.com/file/d/1NC7QHky9NFlRWW-_D43TGxix1vwbfLwK/view?usp=sharing).
- `data_path`: the input .npz file for inference, (in our example `skeletons/testvids_ntu.npz`)

```sh
cd skateformer
python inference.py --config config/sarcopenia/inference
cd ..
```

After running the inference, you can see the results with the **`inference_analysis.py`** script:

```sh
python inference_analysis.py
```

### Training

You first need to download [the skeleton dataset ZIP file.](https://drive.google.com/file/d/1DWhOhUkGeHy8sgf82aMkyJxKeA5oq6us/view?usp=sharing).
Extract this into the directory `skateformer/data/`
The zip contains The skeleton data of the subjects in a pre-split 3-fold data (train/val/test split).
This is for training, validation and testing of the stability classifier on the *stability classification* task.
The ground truth predictions labels are stable/unstable.

```sh
# within the project directory
cd skateformer
python train.py --data_dt data/20251209_1738 --dt train
cd ..
```

This will run the training and testing of the stability classifier on all the partitions,
outputting the training analytics and models (per epoch) to a subdirectory `skateformer/work_dir/train`.
You can then analyze the training runs by:

```sh
# within the project directory
python training_metrics.py skateformer/work_dir/train/
```
