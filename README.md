# Skeleton-Based Motion Pattern Recognition for Sarcopenia Risk Assessment

<img width="500" alt="image" src="https://github.com/user-attachments/assets/3bb581c2-d8bc-46dd-b14f-1e36ccf1f9b2" />
</br>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/8d052410-e9b1-4c6b-96f1-58913a17c7b3" />

Yu-Hsuan Chiu, Aldo Acevedo Onieva, Gee-Sern Hsu, Jiunn-Horng Kang

## Abstract

Sarcopenia, characterized by progressive loss of muscle mass and function, significantly increases fall risk and mobility decline in older adults. Conventional diagnostics require specialized instrumentation and trained personnel, limiting their feasibility for large-scale community screening. We propose a vision-based framework for sarcopenia risk stratification using tandem gait videos captured with a single RGB camera. Tandem gait amplifies balance deficits associated with sarcopenia, making them detectable by vision-based systems. We utilized 124 subjects (93 normal controls, 31 sarcopenia), with tandem gait sequences segmented into skeleton-based clips annotated as stable or unstable. A pretrained spatiotemporal skeleton transformer, SkateFormer, was fine-tuned for stability classification. A sliding-window aggregation approach computed instability rate as a quantitative biomarker for subject-level risk assessment. Risk stratification demonstrated statistically significant separation into mostly stable (14.3\% sarcopenia prevalence) and mostly unstable (46.2\% prevalence) groups. These results validate the feasibility and practical contribution of the proposed method for contact-free, user-friendly, and scalable prescreening of sarcopenia risk.

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

## Dataset
### Data Collection and Ethics

This dataset was collected from 124 community-dwelling older adults (aged 65-80 years) recruited from Taipei Medical University Hospital. The study received institutional review board approval (Taipei Medical University, IRB No. N202203197), and written informed consent was obtained from all participants prior to data collection.

**Sarcopenia diagnosis** followed AWGS 2019 consensus criteria, evaluating muscle mass via BIA (InBody270), muscle strength (handgrip), and physical performance (gait speed or SPPB). The cohort comprised 31 sarcopenia patients and 93 normal controls.

**Video capture protocol**: Tandem gait videos were recorded using consumer-grade smartphones (Samsung Galaxy S9) mounted on tripods at fixed frontal viewpoints (1920×1080 pixels, 29.91 fps). Participants performed standardized heel-to-toe walking, reducing the base of support to amplify balance deficits. Full-length videos (8-15 seconds) were temporally segmented into 3-second clips and annotated by expert raters as stable or unstable.

### Privacy Protection

To protect participant privacy, **only anonymized 3D skeleton data is publicly released**—no raw videos or personally identifiable information are included. Skeleton sequences were extracted using pose estimation algorithms (HRNet + PoseFormerV2), ensuring that individual identities cannot be recovered from the released dataset. All data sharing complies with institutional privacy policies and research ethics standards.

### Released Data Formats

We release the skeleton data in multiple modalities:

- A. 3-second skeleton clips of 124 subjects performing tandem gait labelled stable/unstable and sarcopenia/normal, with subject and time annotations
  - [H36M 17-joint format](https://drive.google.com/file/d/1u-s3XaeU5hLAbFOijnbEbNEiagwPfQua/view?usp=sharing)
  - [H36M 17-joint -> Kinect V2 NTU format](https://drive.google.com/file/d/1cIcbwhk7UwRCYOsq4FJDIRBG6WqsZZUI/view?usp=sharing)
- B. Full-length skeleton clips of 124 subjects performing tandem gait
  - [H36M 17-joint format](https://drive.google.com/file/d/1Gp2eKt1fCNVYBbLejhn9-ZV3k_Y7iIbW/view?usp=sharing)
- C. 3-fold cross-validation ready skeleton clips of 124 subjects performing tandem gait
  - [Kinect V2 NTU format](https://drive.google.com/file/d/1DWhOhUkGeHy8sgf82aMkyJxKeA5oq6us/view?usp=sharing)

## Inference

You need to record videos of the subjects performing tandem gait, and save them into a directory. For example, let's say you have all videos on a directory `/home/user/sarcopenia/testvids`.

If the video is not square, you need to resize it to a square aspect-ratio to make it compatible with the pose estimation model. To facilitate this, we have provided the script **`video_resize.py`**. This script takes all videos from one directory, resizes them to 700x700 (you can adjust this), and saves them to another directory. The method for resizing adds black bars, i.e. it makes the picture fit to the resolution, _not_ fill. This is important, because we don't want to cut-off the subject. Example:

```sh
python video_resize.py /home/user/sarcopenia/testvids /home/user/sarcopenia/testvids_700x700
```

Once you have the resized videos, you need to estimate the poses of the subjects in the videos. For this, we have provided the script **`mp4_to_skeleton.py`**. This script reads video files from a directory and outputs a single `.npz` file containing all the skeletons. You need to place the `models` directory containing the PoseFormerV2, HRNet, YOLO weights required for pose estimation. This is found in [models.zip](https://drive.google.com/file/d/1-HSAujzzb0Xktoty5GPekgyndigxPC-f/view?usp=sharing).

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

We also need to convert to the format SkateFormer accepts for inference:

```sh
python partition_inference.py skeletons/testvids_ntu.npz  skeletons/testvids_inference.npz
```

Now we can stratify the subjects into risk groups depending on their instability rate.
The script `inference.py` does this.
You first must modify the configuration file `config/sarcopenia/inference.yaml`, specifically the lines 5, 6 and 11:

- `work_dir`: output directory of the inference e.g. `work_dir/testvids_inference/`. There will be a CSV containing the sliding window instability rates.
- `weights`: path to the pretrained stability classifier `.pt` file. You can download this [here](https://drive.google.com/file/d/1NC7QHky9NFlRWW-_D43TGxix1vwbfLwK/view?usp=sharing).
- `data_path`: the input .npz file for inference, (e.g. `../skeletons/testvids_inference.npz`)

```sh
cd skateformer
python inference.py --config config/sarcopenia/inference.yaml
cd ..
```

After running the inference, you can see the results with the **`inference_analysis.py`** script:

```sh
python inference_analysis.py skateformer/work_dir/testvids_inference
```

## Training

You need to download the [3-fold cross-validation ready version of the skeleton dataset.](#dataset)
Extract the zip into the directory `skateformer/data/`.
The zip contains the skeleton data of the subjects in a pre-configured 3-fold cross-validation format (train/val/test split).
This is for training, validation and testing of the stability classifier on the *stability classification* task.
The ground truth predictions labels are stable/unstable.

You also need the pretrained SkateFormer model, which we will finetune.
It must be located at `./pretrained/ntu120_CSub/SkateFormer_j.pt`.
You can download from the [SkateFormer repository.](https://github.com/KAIST-VICLab/SkateFormer?tab=readme-ov-file#pretrained-model)

To train the models, run:

```sh
# within the project directory
cd skateformer
python train.py --data_dt 20251209_1738 --dt train
cd ..
```

This will run the training and testing of the stability classifier on all the partitions,
outputting the training analytics and models (per epoch) to a subdirectory `skateformer/work_dir/train`.
You can then analyze the training runs by:

```sh
# within the project directory
python training_metrics.py skateformer/work_dir/train/
```

