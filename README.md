# Semantic Adversarial Attacks via Diffusion Models [BMVC 2023]

[arXiv](https://arxiv.org/abs/2309.07398)

This is the official implementation of the paper "Semantic Adversarial Attacks via Diffusion Models".
This codebase is built on top of [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP) (mainly) and [LatentHSJA](https://github.com/ndb796/LatentHSJA) (the dataset and the classifier).
The proposed ST and LM approaches are implemented in this codebase.

### Other Git Repositories Used

[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for explainable models.
[pytorch-fid](https://github.com/mseitzer/pytorch-fid) for calculating fid scores.
[gan-metrics-pytorch](https://github.com/abdulfatir/gan-metrics-pytorch) for calculating kid scores.
[improved-precision-and-recall-metric-pytorch](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch) for calculating precision and recall.
We appreciate the efforts made from the opensource community, if some libraries we used and not mentioned here, feel free to let us know!

### Environment Setup

The enviroment we use is very similar to [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP), and we also provide a `environments.yml` file. You can run the following commands to setup environment:

```shell
conda env create --name envname --file=environments.yml
```

Note that we use pytorch==1.11.0 and Python==3.8.12.

### Prepare Datasets

The CelebA-HQ Facial Identity Recognition Dataset and CelebA-HQ Face Gender Recognition Dataset can be downloaded from [LatentHSJA](https://github.com/ndb796/LatentHSJA).
The AFHQ dataset can be downloaded from [StarGanv2](https://github.com/clovaai/stargan-v2).

### Training Victim Classifier

For CelebA-HQ identity dataset, please refer to [this](https://github.com/ndb796/LatentHSJA/blob/main/classification_models/Face_Gender_Classification_Using_Transfer_Learning_with_ResNet18_Resolution_256_Normalize_05.ipynb).
For CelebA-HQ gender dataset, plrease refer to [this](https://github.com/ndb796/LatentHSJA/blob/main/classification_models/Facial_Identity_Classification_Using_Transfer_Learning_with_ResNet18_Resolution_256_Normalize_05.ipynb).
The model weights can be downloaded from [LatentHSJA](https://github.com/ndb796/LatentHSJA).
You could also run the following for CelebA-HQ identity and gender datasets (change dataset dir for gender dataset):

```shell
python train_classifier.py
```

and the following for AFHQ dataset:

```shell
python train_classifier_afhq.py
```

### Obtain Indices of Attackable Images

Only correctly classified images are attackable. Also, in the LM approach, we must find image pairs with predicted labels.
The following commands are for CelebA-HQ identity dataset, CelebA-HQ gender dataset, and AFHQ dataset, respectively:

```shell
python pick_indices_identity.py  # pick indices for CelebA-HQ identity dataset
python pick_indices_gender.py 	 # pick indices for CelebA-HQ gender dataset
python pick_indices_afhq.py 	 # pick indices for AFHQ dataset
```

The indices would be saved as pickle files. These indices is for a specific classifier (i.e. the victim classifier).

### Prepare Diffusion Models

For CelebA-HQ identity and gender datasets, we use the diffusion model weights pretrained on CelebA-HQ from [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP): [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view).
For AFHQ dataset, we use the diffusion model weights from [ILVR+ADM](https://github.com/jychoi118/ilvr_adm):[drive](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&id=72419B431C262344%21103807&cid=72419B431C262344).
and finetune it with the following (although the best way is to train a diffusion model for AFHQ dataset from scratch):

```shell
python main_afhq_train.py
```

### Generate Attacks

The commands for CelebA-HQ identity dataset are stored in `commands/command_for_celebaHQ_identity_ST_approach` and `commands/command_for_celebaHQ_identity_LM_approach` folders.

The commands for CelebA-HQ gender dataset are stored in `commands/command_for_celebaHQ_gender` folder.

The commands for AFHQ dataset are stored in `commands/command_for_AFHQ` folder.

Basically, taking CelebA-HQ identity dataset as an example, for the ST approach, we have:

```shell
python main.py --attack                 \
            --config celeba.yml         \
            --exp experimental_log_path \
            --t_0 500                   \
            --n_inv_step 40             \
            --n_test_step 40            \
            --n_precomp_img 100 --mask 9 --diff 9 --tune 0 --black 0
```

parameters:

- `--config` specifies which dataset we use
- `--exp` specifies the log directory
- `--t_0` specifies how many semantic adversarial images we'd like to generate
- `--black` specifies black/white-box attack (0 for white-box and 1 for black-box)
- `--tune` is the attack strategy (0 for fintuning the latent space, 1 for finetuning the diffusion model, and 2 for finetuning both the latent space and diffusion model).
- Some parameters are unused, e.g. `--mask` and `--diff` since these are for the LM approach.

Taking CelebA-HQ identity dataset as an example, for the LM approach, we have:

```shell
python main.py --attack                 \
            --config celeba.yml         \
            --exp experimental_log_path \
            --n_test_img 4              \
            --t_0 500                   \
            --n_inv_step 40             \
            --n_test_step 40            \
            --n_precomp_img 500 --mask 2 --diff 1 --tune 3 --black 0
```

- `--config` specifies which dataset we use
- `--exp` specifies the log directory
- `--t_0` specifies how many semantic adversarial images we'd like to generate
- `--black` specifies black/white-box attack (0 for white-box and 1 for black-box)
- `--mask` specifies which explainable model we use (2 for GradCAM, 3 for FullGrad, 4 for SimpleFullGrad, and 5 for SmoothFullGrad)
- `--diff` specifies based on which we generate the attack (1 for source image, 2 for target image, 3 for combing both source and target image)
- `--tune` should be set to 3 for all cases in the LM approach.

For CelebA-HQ gender dataset and AFHQ dataset, the only difference is to replace the dataset and the pretrained diffusion model as in the `command_for_CelebaHQ_gender` and `command_for_AFHQ` folders.

Please cite our paper if you feel this is helpful:

```
@inproceedings{wang2023semantic,
  title={Semantic Adversarial Attacks via Diffusion Models},
  author={Wang, Chenan and Duan, Jinhao and Xiao, Chaowei and Kim, Edward and Stamm, Matthew and Xu, Kaidi},
  booktitle={34rd British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24},
  year={2023}
}
```
