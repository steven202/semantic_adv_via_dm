import datetime
import gc
import pickle
import sys
import time
from torchvision.models.inception import inception_v3
from glob import glob
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
import torchvision.transforms as tfs
from torchvision import datasets, transforms
import torchvision
from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.smooth_fullgrad import SmoothFullGrad
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import (
    DATASET_PATHS,
    MODEL_PATHS,
    HYBRID_MODEL_PATHS,
    HYBRID_CONFIG,
)
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.optim as optim
import torch.nn.functional as F
import torchattacks
import lpips
import time


class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.cpu = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == "fixedsmall":
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def attack(self):
        # ----------- Models -----------#
        print(self.args.exp)
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        models = []
        model_paths = [None, None]
        all_step_num = [
            int(tmp.split("_")[1])
            for tmp in os.listdir(self.args.image_folder)
            if "_only_reconstruction.pt" in tmp
        ]
        if len(all_step_num) > 0:
            latest = np.max(all_step_num)
            self.args.resume_step = latest + 1
        else:
            self.args.resume_step = 0
        if self.args.resume_step >= self.args.n_precomp_img:
            sys.exit()
        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(
                        url, map_location=self.device
                    )
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                model_i = i_DDPM(self.config.data.dataset)
                if self.config.data.dataset == "AFHQ" and model_path == None:
                    model_path = "pretrained/afhq_dog_4m.pt"
                    with open(
                        os.path.join(self.args.image_folder, f'{"train"}_log.txt'), "a+"
                    ) as file1:
                        file1.write(f"load afhq model path")
                    print("load afhq model path")
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print("Not implemented dataset")
                raise ValueError
            self.ckpt = ckpt
            model_i.load_state_dict(ckpt)
            model_i.to(self.device)
            model_i = torch.nn.DataParallel(model_i)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)
        ##### loading dataset ############
        train_transform = tfs.Compose(
            [
                transforms.Resize((256, 256)),
                # transforms.RandomHorizontalFlip(),  # data augmentation
                tfs.ToTensor(),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        test_transform = tfs.Compose(
            [
                transforms.Resize((256, 256)),
                tfs.ToTensor(),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        if self.config.data.dataset == "CelebA_HQ" and self.args.face == "identity":
            data_dir = "./CelebA_HQ_facial_identity_dataset"
            train_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "train"), train_transform
            )
            test_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "test"), test_transform
            )
            save_path = "facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"
            desc = "face identity classification"
            with open("saved_indices/saved_indices.pkl", "rb") as f:
                saved_indices = pickle.load(f)

        elif self.config.data.dataset == "CelebA_HQ" and self.args.face == "gender":
            data_dir = "./CelebA_HQ_face_gender_dataset"
            train_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "train"), train_transform
            )
            test_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "test"), test_transform
            )
            save_path = "models_celebhq_gender/face_gender_classification_using_transfer_learning_with_ResNet18_resolution_256_normalize_05.pth"
            desc = "face gender classification"
            with open("saved_indices/saved_indices_gender.pkl", "rb") as f:
                saved_indices = pickle.load(f)
        elif self.config.data.dataset == "AFHQ":
            data_dir = "data/afhq"
            train_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "train"), train_transform
            )
            test_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "val"), test_transform
            )
            save_path = "afhq_resnet18.pth"
            desc = "animal classification"
            with open("saved_indices/saved_indices_afhq_dog.pkl", "rb") as f:
                saved_indices = pickle.load(f)
        with open(
            os.path.join(self.args.image_folder, f'{"train"}_log.txt'), "a+"
        ) as file1:
            file1.write("==================" + desc + "\n")
            print(desc)
        # loader_dic = get_dataloader(
        #     train_dataset,
        #     test_dataset,
        #     bs_train=2,
        #     num_workers=self.config.data.num_workers,
        # )
        # loader = loader_dic["train"]
        source_index_train = saved_indices["source_index_train"]
        target_index_train = saved_indices["target_index_train"]
        source_index_test = saved_indices["source_index_test"]
        target_index_test = saved_indices["target_index_test"]
        # assert len(source_index_test) >= 500 and len(target_index_test) == len(source_index_test)

        if False:
            net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
            net.heads = nn.Sequential(nn.Linear(768, 307))
            net = nn.Sequential(nn.AdaptiveAvgPool2d((224, 224)), net)
            save_path = "classifier_ckpts/facial_identity_classification_transfer_learning_with_ViT_resolution_256_5.pth"
            net.load_state_dict(torch.load(save_path, map_location="cuda"))
            net.eval()
        elif False:
            net = torchvision.models.resnet18(pretrained=True)
            num_features = net.fc.in_features
            net.fc = nn.Linear(num_features, len(train_dataset.classes))
            net = net.to(self.device)
            net = nn.DataParallel(net)
            save_path = "classifier_ckpts/resnet_pgd_l2.pt"
            net.load_state_dict(torch.load(save_path, map_location="cuda")["net"])
            net.eval()
        elif True:
            net = torchvision.models.resnet18(pretrained=True)
            num_features = net.fc.in_features
            net.fc = nn.Linear(num_features, len(train_dataset.classes))
            net = net.to(self.device)
            net = nn.DataParallel(net)
            save_path = "classifier_ckpts/resnet_pgd_linf.pt"
            net.load_state_dict(torch.load(save_path, map_location="cuda")["net"])
            net.eval()
        else:
            net = torchvision.models.resnet18(pretrained=True)
            num_features = net.fc.in_features
            net.fc = nn.Linear(num_features, len(train_dataset.classes))
            net = net.to(self.device)
            net.load_state_dict(torch.load(save_path, map_location="cuda"))
            net.eval()
        if (
            self.args.mask == 3
            or self.args.mask == 4
            or self.args.mask == 5
            or self.args.mask == 7
        ):
            if self.args.mask == 3:
                fullgrad = FullGrad(net)
                fullgrad.checkCompleteness()
            elif self.args.mask == 4:
                fullgrad = SimpleFullGrad(net)
            elif self.args.mask == 5:
                fullgrad = SmoothFullGrad(net)
            elif self.args.mask == 7:
                fullgrad = SmoothFullGrad(
                    torchvision.models.resnet50(pretrained=True).to(self.device)
                )
        elif self.args.mask == 2:
            target_layers = [net.layer4[-1]]
            cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        delta_times = []

        count = 0
        total = 0
        targeted = 0
        ratio_list = []
        query_list = []
        if self.args.black == 1:
            self.inception_model = (
                inception_v3(pretrained=True, transform_input=False)
                .type(torch.FloatTensor)
                .to(self.device)
            )
            self.inception_model.fc = torch.nn.Identity()
            self.inception_model.eval()
        starting = time.time()
        mode = "train"

        # self.args.resume_step = args.self.args.resume_step
        for step in range(0, len(source_index_test)):
            if step < self.args.resume_step:
                continue
            time_in_start = time.time()
            net = net.to(self.device)
            net.eval()
            source_inputs, source_labels = train_dataset[source_index_train[step]]
            target_inputs, target_labels = train_dataset[target_index_train[step]]
            # source_inputs, target_inputs = source_inputs.unsqueeze(0), target_inputs.unsqueeze(0)
            images = torch.stack([source_inputs, target_inputs])
            labels = torch.tensor([source_labels, target_labels]).to(torch.long)
            # prepare first image
            img = images[0].unsqueeze(dim=0)
            x0 = img.to(self.config.device)
            x_0 = x0.detach().clone()
            label1 = labels[0].detach().clone().cuda()
            x_1 = images[0].detach().clone().unsqueeze(dim=0).cuda()

            with torch.no_grad():
                output1 = net(x_1)
                pred1 = output1.max(dim=1)[1].item()
                original_logit = nn.Softmax(dim=1)(output1)[0, label1.item()].item()
                if self.args.tune == 3 or self.args.tune == 4:
                    # prepare second image
                    img3 = images[1].unsqueeze(dim=0)
                    x3 = img3.to(self.config.device)
                    x_3 = x3.detach().clone()
                    label2 = labels[1].detach().clone().cuda()
                    x_2 = images[1].detach().clone().unsqueeze(dim=0).cuda()
                    # calculate output
                    output2 = net(x_2)
                    pred2 = output2.max(dim=1)[1].item()
                    if (
                        pred1 == pred2
                        or pred1 != label1.item()
                        or pred2 != label2.item()
                    ):
                        continue
                    ############
                    # tvu.save_image((x_1 + 1) * 0.5, f"./original_500_1/{step}_{pred1}.png")
                    # tvu.save_image((x_2 + 1) * 0.5, f"./original_500_2/{step}_{pred2}.png")
                    # if step >= 500:
                    #     break
                    # continue
                    #############
            if self.args.tune == 3 or self.args.tune == 4:
                if self.args.mask == 2:
                    target1 = [ClassifierOutputTarget(label1)]
                    target2 = [ClassifierOutputTarget(label2)]
                    diff_1 = (
                        torch.from_numpy(cam(input_tensor=x_1, targets=target1))
                        .repeat(1, 3, 1, 1)
                        .cuda()
                    )
                    diff_2 = (
                        torch.from_numpy(cam(input_tensor=x_2, targets=target2))
                        .repeat(1, 3, 1, 1)
                        .cuda()
                    )
                elif (
                    self.args.mask == 3
                    or self.args.mask == 4
                    or self.args.mask == 5
                    or self.args.mask == 7
                ):
                    diff_1 = fullgrad.saliency(x_1, label1).repeat(1, 3, 1, 1).cuda()
                    diff_2 = fullgrad.saliency(x_2, label2).repeat(1, 3, 1, 1).cuda()
                if self.args.diff == 1:
                    mask_3 = torch.abs(diff_1)
                elif self.args.diff == 2:
                    mask_3 = torch.abs(diff_2)
                elif self.args.diff == 3:
                    mask_3 = torch.abs(torch.abs(diff_1) + torch.abs(diff_2))
                else:
                    raise Exception("not supported diff")
                mask_3 /= mask_3.max()
            ################# generate two latent spaces ####################
            # clean_latents = [x_0.detach().clone()]
            # target_latents = [x_3.detach().clone()]
            # clean_latents, target_latents = [], []
            # clean_reverses = []
            # target_reverses = []
            with torch.no_grad():
                with tqdm(
                    total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                ) as progress_bar:
                    for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_prev = (torch.ones(n) * j).to(self.device)
                        x_0 = denoising_step(
                            x_0,
                            t=t,
                            t_next=t_prev,
                            models=models,
                            logvars=self.logvar,
                            sampling_type="ddim",
                            b=self.betas,
                            eta=0,
                            learn_sigma=learn_sigma,
                            ratio=0,
                        )
                        # clean_latents.append(x_0.detach().clone())
                        progress_bar.update(1)
                        x_0_tmp = x_0.detach().clone()
                        del t, t_prev, x_0
                        gc.collect()
                        torch.cuda.empty_cache()
                        x_0 = x_0_tmp
                x_lat = x_0.detach().clone()
                if self.args.tune == 3 or self.args.tune == 4:
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x_3 = denoising_step(
                                x_3,
                                t=t,
                                t_next=t_prev,
                                models=models,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                                ratio=0,
                            )
                            # target_latents.append(x_3.detach().clone())
                            progress_bar.update(1)
                            x_3_tmp = x_3.detach().clone()
                            del t, t_prev, x_3
                            gc.collect()
                            torch.cuda.empty_cache()
                            x_3 = x_3_tmp

                    x_lat__ = x_3.detach().clone()
                    self.mask_3 = mask_3.detach().to(self.device)
                else:
                    x_lat__ = None
                    x3 = None
            ########## start to find the best threshold #########################
            gc.collect()
            torch.cuda.empty_cache()
            # x_lat_0 = x_0.detach().clone()
            # img_lat_pair = [x_1.detach().clone(), None, x_lat_0.detach().clone()]
            # img_lat_pairs_dic = [x_0.detach().clone(), None, x_lat_0.detach().clone()]
            # eps = 1e-14  # 16.0
            # mask3_channels = mask_3.squeeze(0)
            query_list.append(0)
            self.ratio = 100
            # query_list[-1] += 1
            # self.mask_4 = self.prune_mask(self.mask_3)
            x_generate, pred3 = self.generate_attack(
                models,
                learn_sigma,
                net,
                seq_inv,
                seq_inv_next,
                n,
                mode,
                step,
                x0,
                x3,
                label1,
                x_lat,
                x_lat__,
                query_list,
                original_logit,
            )
            assert not ((x_generate > 1).any() or (x_generate < -1).any())
            ratio_list.append(self.ratio)
            tvu.save_image(
                (x_generate + 1) * 0.5,
                os.path.join(
                    self.args.image_folder,
                    f"{mode}_{step}_{self.ratio}_only_reconstruction.png",
                ),
            )
            # image = train_transform(Image.open(os.path.join(self.args.image_folder,f"{mode}_{step}_{self.ratio}_only_reconstruction.png",)).convert("RGB")).unsqueeze(0)
            torch.save(
                x_generate,
                os.path.join(
                    self.args.image_folder,
                    f"{mode}_{step}_{self.ratio}_only_reconstruction.pt",
                ),
            )
            tvu.save_image(
                (x0 + 1) * 0.5,
                os.path.join(
                    self.args.image_folder,
                    f"{mode}_{step}_{self.ratio}_only_original.png",
                ),
            )
            torch.save(
                x0,
                os.path.join(
                    self.args.image_folder,
                    f"{mode}_{step}_{self.ratio}_only_original.pt",
                ),
            )
            if self.args.tune == 3 or self.args.tune == 4:
                tvu.save_image(
                    (x3 + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder,
                        f"{mode}_{step}_{self.ratio}_only_target.png",
                    ),
                )
                torch.save(
                    x3,
                    os.path.join(
                        self.args.image_folder,
                        f"{mode}_{step}_{self.ratio}_only_target.pt",
                    ),
                )
            if self.args.tune == 3 or self.args.tune == 4:
                mask_to_save = self.mask_4
            else:
                mask_tmp = torch.abs(x_generate - x0)
                mask_tmp /= mask_tmp.max()
                mask_to_save = (mask_tmp + 1) * 0.5
            tvu.save_image(
                mask_to_save,
                os.path.join(
                    self.args.image_folder, f"{mode}_{step}_{self.ratio}_only_mask.png"
                ),
            )
            if self.args.tune == 3 or self.args.tune == 4:
                image_lst = [
                    (x0 + 1) * 0.5,
                    (x3 + 1) * 0.5,
                    mask_to_save,
                    (x_generate + 1) * 0.5,
                ]
            else:
                image_lst = [
                    (x0 + 1) * 0.5,
                    mask_to_save,
                    (x_generate + 1) * 0.5,
                ]
            grid_images = torchvision.utils.make_grid(
                torch.cat(
                    image_lst,
                    dim=0,
                ),
                nrow=8,
            )
            if self.args.tune == 3 or self.args.tune == 4:
                tvu.save_image(
                    grid_images,
                    os.path.join(
                        self.args.image_folder,
                        f'{mode}_{step}_{self.ratio}_{"true" if pred1!=pred3 else "false"}_lbl1_{label1.item()}_lbl2_{label2.item()}_pred3_{pred3}_1_rec.png',
                    ),
                )
            else:
                tvu.save_image(
                    grid_images,
                    os.path.join(
                        self.args.image_folder,
                        f'{mode}_{step}_{self.ratio}_{"true" if pred1!=pred3 else "false"}_lbl1_{label1.item()}_pred3_{pred3}_1_rec.png',
                    ),
                )
            count += 1 if pred1 != pred3 else 0
            if self.args.tune == 3 or self.args.tune == 4:
                targeted += 1 if (pred3 == label2.item() and pred1 != pred3) else 0
            total += 1
            print(
                f"Total count: {mode}_step_{step}_avr_{np.mean(ratio_list)}_asr_{count*1.0/total}_atr_{targeted*1.0/total}_targeted_{targeted}_untargeted_{count}_total_{total}_avgquery_{np.mean(query_list)}_avgtime_{((time.time()-starting)*1.0/total):.4f}_totquery_{np.sum(query_list)}_tottime_{((time.time()-starting)*1.0):.4f}\n"
            )
            with open(
                os.path.join(self.args.image_folder, f"{mode}_log.txt"), "a+"
            ) as file1:
                if self.args.tune == 3 or self.args.tune == 4:
                    file1.write(
                        f'Per image: {mode}_{step}_{self.ratio}_{"true" if pred1!=pred3 else "false"}_avr_{np.mean(ratio_list)}_lbl1_{label1.item()}_lbl2_{label2.item()}_pred3_{pred3}_1_rec\n'
                    )
                else:
                    file1.write(
                        f'Per image: {mode}_{step}_{self.ratio}_{"true" if pred1!=pred3 else "false"}_avr_{np.mean(ratio_list)}_lbl1_{label1.item()}_pred3_{pred3}_1_rec\n'
                    )
                file1.write(
                    f"Total count: {mode}_step_{step}_avr_{np.mean(ratio_list)}_asr_{count*1.0/total}_atr_{targeted*1.0/total}_targeted_{targeted}_untargeted_{count}_total_{total}_avgquery_{np.mean(query_list)}_avgtime_{((time.time()-starting)*1.0/total):.4f}_totquery_{np.sum(query_list)}_tottime_{((time.time()-starting)*1.0):.4f}\n"
                )
                rest_time_estimate_in_sec = (
                    self.args.n_precomp_img - (total + self.args.resume_step)
                ) * ((time.time() - starting) * 1.0 / total)
                rest_time_estimate = str(
                    datetime.timedelta(seconds=rest_time_estimate_in_sec)
                )
                file1.write(f"rest_time_estimate: {rest_time_estimate}\n")
            time_in_end = time.time()
            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
            delta_times.append(time_in_end - time_in_start)
            print(f"average elapsed time: {np.mean(delta_times):.4f}")
            print(f"rest_time_estimate: {rest_time_estimate}\n")
            if total + self.args.resume_step >= self.args.n_precomp_img:
                print(f"Forall: average elapsed time: {np.mean(delta_times):.4f}")
                break

    def get_x_train(self, x_lat, x_lat__):
        return x_lat__ * self.mask_4 + x_lat * (1 - self.mask_4)

    def generate_attack(
        self,
        models,
        learn_sigma,
        net,
        seq_inv,
        seq_inv_next,
        n,
        mode,
        step,
        x0,
        x3,
        label1,
        x_lat,
        x_lat__=None,
        query_list=None,
        original_logit=None,
    ):
        new_logit = None
        output_logits = None
        pred1 = label1.item()
        # ----------- Optimizer and Scheduler -----------#
        model = models[1]
        model.module.load_state_dict(self.ckpt)
        model.to(self.device)
        if self.args.tune == 3 or self.args.tune == 4:
            self.mask_4 = self.prune_mask(self.mask_3)
        # x_train = x_lat__ * mask_4 + x_lat * (1 - mask_4)
        # x_train = x_lat
        # x_train0 = x_lat.detach().clone().to(x_lat.device)
        # x_train = x_lat.detach().clone().to(x_lat.device).requires_grad_()

        # x_train = x_lat__ * mask_4 + x_lat * (1 - mask_4)
        # x_train = self.get_x_train(x_lat, x_lat__)

        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        if self.args.tune == 0:  # finetune latent
            x_train = x_lat.detach().clone().to(x_lat.device).requires_grad_()
            if self.args.train == 0:
                optim_ft = torch.optim.Adam(
                    [x_train], weight_decay=0, lr=self.args.lr_clip_finetune
                )
            else:
                optim_ft = torch.optim.Adam(
                    [
                        {"params": net.parameters()},
                        {"params": x_train},
                    ],
                    weight_decay=0,
                    lr=self.args.lr_clip_finetune,
                )
        elif (
            self.args.tune == 3 or self.args.tune == 4
        ):  # finetune mask with another image or random noise
            # if self.args.train == 0:
            #     optim_ft = torch.optim.Adam([self.mask_4], weight_decay=0, lr=self.args.lr_clip_finetune)
            # else:
            #     optim_ft = torch.optim.Adam(
            #         [
            #             {"params": net.parameters()},
            #             {"params": self.mask_4},
            #         ],
            #         weight_decay=0,
            #         lr=self.args.lr_clip_finetune,
            #     )
            if self.args.tune == 3:
                x_train = self.get_x_train(x_lat, x_lat__)
            elif self.args.tune == 4:
                x_train = self.get_x_train(
                    x_lat, torch.rand(*x_lat.shape, device=x_lat.device)
                )
        elif self.args.tune == 1:  # finetune model
            x_train = x_lat.detach().clone().to(x_lat.device).requires_grad_()
            optim_ft = torch.optim.Adam(
                model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune
            )
        elif self.args.tune == 2:  # finetune latent and model
            x_train = x_lat.detach().clone().to(x_lat.device).requires_grad_()
            optim_ft = torch.optim.Adam(
                [
                    {"params": model.parameters()},
                    {"params": x_train},
                ],
                weight_decay=0,
                lr=self.args.lr_clip_finetune,
            )

        # ----------- Loss -----------#
        print("Loading losses")
        loss_fn_alex = lpips.LPIPS(net="alex").to(self.device)
        id_loss_func = id_loss.IDLoss().to(self.device).eval()
        kl_loss = nn.KLDivLoss(reduction="sum").to(self.device)
        mse_loss = nn.MSELoss().to(self.device)
        # delta # misclassification
        # ----------- Finetune Diffusion Models -----------#
        if self.args.tune != 3 and self.args.tune != 4:
            init_opt_ckpt = optim_ft.state_dict()
            scheduler_ft = torch.optim.lr_scheduler.StepLR(
                optim_ft, step_size=1, gamma=self.args.sch_gamma
            )
            init_sch_ckpt = scheduler_ft.state_dict()
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)

        model.module.load_state_dict(self.ckpt)

        # finetune process start
        print("Start finetuning")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        self.args.n_train_step = 15  # ???
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print("Uniform skip type")
        else:
            seq_train = list(range(self.args.t_0))
            print("No skip")
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        gc.collect()
        torch.cuda.empty_cache()
        # x_generate = x_lat__ * mask_4 + x_lat * (1 - mask_4)
        # ----------- Train -----------#
        self.args.n_iter = 30
        finetine_iterations = 1
        for it_out in range(self.args.n_iter):
            query_list[-1] += 1
            model.train()
            if self.args.black == 0:
                net.train()
            for step in range(finetine_iterations):
                # x_train = x_lat_train.detach().clone()
                if self.args.tune == 3 or self.args.tune == 4:
                    if self.args.tune == 3:
                        self.ratio -= self.find_threshold_decremental(
                            original_logit, new_logit, output_logits, label1
                        )
                        if self.ratio <= 0:
                            self.ratio = 0
                        self.mask_4 = self.prune_mask(self.mask_3)
                        x_train = self.get_x_train(x_lat, x_lat__)  # ???
                    elif self.args.tune == 4:
                        self.ratio -= self.find_threshold_decremental(
                            original_logit, new_logit, output_logits, label1
                        )
                        if self.ratio <= 0:
                            self.ratio = 0
                        self.mask_4 = self.prune_mask(self.mask_3)
                        x_train = self.get_x_train(
                            x_lat, torch.rand(*x_lat.shape, device=x_lat.device)
                        )
                    # no finetune, evaluate it directly
                    # self.cpu = torch.device("cuda")
                    x_eval, pred4, new_logit, output_logits = self.evaluate(
                        learn_sigma,
                        net,
                        n,
                        model,
                        x_train,
                        seq_test,
                        seq_test_next,
                        label1,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if pred4 != pred1:
                        return x_eval.detach(), pred4
                    elif self.ratio == 0:
                        return x_eval.detach(), pred4
                    else:
                        continue

                optim_ft.zero_grad()
                with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                    for t_it, (i, j) in enumerate(
                        zip(reversed(seq_train), reversed(seq_train_next))
                    ):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        if t_it == 0:
                            x_train_ = denoising_step(
                                x_train,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                eta=self.args.eta,
                                learn_sigma=learn_sigma,
                            )
                        else:
                            x_train_ = denoising_step(
                                x_train_,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                eta=self.args.eta,
                                learn_sigma=learn_sigma,
                            )
                        del t, t_next
                        gc.collect()
                        torch.cuda.empty_cache()
                        progress_bar.update(1)
                        # doing this in the inner loop for blackbox ST attack boost ASR, but lose fidelity.
                        if self.args.tune == 0 and self.args.black == 1:
                            x_train_.clamp_(min=-1.0, max=1.0)
                if not (self.args.tune == 0 and self.args.black == 1):
                    x_train_.clamp_(min=-1.0, max=1.0)
                loss_lilps = loss_fn_alex(x0, x_train_)
                loss_id = torch.mean(id_loss_func(x0, x_train_))
                loss_l1 = nn.L1Loss()(x0, x_train_)

                if self.args.black == 0:
                    loss = (
                        1 * loss_lilps
                        # +3 * mse_loss(x_train_, x0)
                        + 0 * loss_id
                        + 0 * loss_l1
                        - 1
                        * kl_loss(
                            F.log_softmax(net(x_train_), dim=1),
                            F.softmax(net(x0), dim=1),
                        )
                    )
                    # if self.args.tune == 3 or self.args.tune == 4:
                    #     loss += self.mask_4.sum()
                else:
                    loss = (
                        +1 * loss_lilps
                        + 0 * loss_id
                        + 0 * loss_l1
                        # + 1.0 * mse_loss(x_train_, x0)
                        - 1
                        * kl_loss(
                            F.log_softmax(self.inception_model(x_train_), dim=1),
                            F.softmax(self.inception_model(x0), dim=1),
                        )
                    )
                loss.backward()
                # x_train.grad.top_k()
                # other grad set =0
                optim_ft.step()
                for p in model.module.parameters():
                    p.grad = None
                gc.collect()
                torch.cuda.empty_cache()
                # evaluation
                net.eval()
                with torch.no_grad():
                    output_logits = net.to(self.cpu)(x_train_.to(self.cpu))
                    pred3 = output_logits.max(dim=1)[1].item()
                    new_logit = nn.Softmax(dim=1)(output_logits)[
                        0, label1.item()
                    ].item()
                    net.to(self.device)
                gc.collect()
                torch.cuda.empty_cache()
                if pred3 != pred1:
                    x_eval, pred4, new_logit, output_logits = self.evaluate(
                        learn_sigma,
                        net,
                        n,
                        model,
                        x_train,
                        seq_test,
                        seq_test_next,
                        label1,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    if pred4 != pred1:
                        return x_eval, pred4
            if self.args.tune != 3 and self.args.tune != 4:
                scheduler_ft.step()
        if self.args.tune == 3 or self.args.tune == 4:
            if self.args.tune == 3:
                self.mask_4 = self.prune_mask(self.mask_3)
                x_train = self.get_x_train(x_lat, x_lat__)
            elif self.args.tune == 4:
                self.mask_4 = self.prune_mask(self.mask_3)
                x_train = self.get_x_train(
                    x_lat, torch.rand(*x_lat.shape, device=x_lat.device)
                )
        x_eval, pred4, new_logit, output_logits = self.evaluate(
            learn_sigma, net, n, model, x_train, seq_test, seq_test_next, label1
        )
        gc.collect()
        torch.cuda.empty_cache()
        return x_eval, pred4

    def evaluate(
        self, learn_sigma, net, n, model, x_train, seq_test, seq_test_next, label1
    ):
        model.eval()
        net.eval()
        x = x_train.detach().clone().requires_grad_(False).to(self.cpu)
        with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
            for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                t = (torch.ones(n) * i).to(self.cpu)
                t_next = (torch.ones(n) * j).to(self.cpu)

                x = denoising_step(
                    x,
                    t=t,
                    t_next=t_next,
                    models=model.module.to(self.cpu),
                    logvars=self.logvar,
                    sampling_type=self.args.sample_type,
                    b=self.betas,
                    eta=self.args.eta,
                    learn_sigma=learn_sigma,
                )

                progress_bar.update(1)

                x_tmp = x.detach().clone()
                del t, t_next, x
                gc.collect()
                torch.cuda.empty_cache()
                x = x_tmp
                # x.clamp_(min=-1.0, max=1.0)
                # doing this in the inner loop for blackbox ST attack boost ASR, but lose fidelity.
                if self.args.tune == 0 and self.args.black == 1:
                    x.clamp_(min=-1.0, max=1.0)
        if not (self.args.tune == 0 and self.args.black == 1):
            x.clamp_(min=-1.0, max=1.0)
        with torch.no_grad():
            output4 = net.to(self.cpu)(x.to(self.cpu))
            pred4 = output4.max(dim=1)[1].item()
            new_logit = nn.Softmax(dim=1)(output4)[0, label1.item()].item()
        gc.collect()
        torch.cuda.empty_cache()
        net.to(self.device)
        model.module.to(self.device)
        return x.to(self.device), pred4, new_logit, output4

    def prune_mask(self, mask_3):
        mask_4 = mask_3.detach().clone()
        with torch.no_grad():
            mask_4 /= mask_4.max()
        if True:
            if True:
                # sum all channels, and prune all channels
                mask3_single_channel = mask_4.squeeze(0).sum(dim=0)
                H, W = mask3_single_channel.shape
                num_k = int((1 - self.ratio / 100.0) * mask3_single_channel.numel())
                tmp = mask3_single_channel.view(-1)
                print("ratio:", self.ratio, "num_k", num_k, "total:", tmp.numel())
                values, index = tmp.topk(num_k)
                two_d_indices = torch.cat(
                    ((index // W).unsqueeze(1), (index % W).unsqueeze(1)), dim=1
                )
                mask_placeholder = torch.zeros(mask_4.shape)
                mask_placeholder[0, :, two_d_indices[:, 0], two_d_indices[:, 1]] = 1
            elif True:
                # prune 3 channels separately
                mask_placeholder = torch.zeros(mask_4.shape)
                for channel in range(3):
                    mask3_single_channel = mask_4.squeeze(0).sum(dim=0)
                    # mask3_single_channel = mask_4.squeeze(0)[channel,:,:]
                    H, W = mask3_single_channel.shape
                    num_k = int((1 - self.ratio / 100.0) * mask3_single_channel.numel())
                    tmp = mask3_single_channel.view(-1)
                    values, index = tmp.topk(num_k)
                    two_d_indices = torch.cat(
                        ((index // W).unsqueeze(1), (index % W).unsqueeze(1)), dim=1
                    )
                    mask_placeholder[
                        0, channel, two_d_indices[:, 0], two_d_indices[:, 1]
                    ] = 1
            with torch.no_grad():
                mask_4[mask_placeholder.le(0.5)] = 0.0
                # mask_4[mask_placeholder.gt(0.5)] = 1.0
        else:
            threshold_3 = torch.quantile(mask_4.flatten(), self.ratio / 100.0)
            mask_4[mask_4 < threshold_3] = 0

        # elif self.args.renorm == 2:
        # mask_4[mask_placeholder.gt(0.5)] = 1
        mask_4 = mask_4.to(self.device).requires_grad_()
        return mask_4

    # original_logit: the probability on targeted label with the original clean image
    # new_logit: the probability on targeted label with the new adversarial image
    def find_threshold_decremental(
        self, original_logit, new_logit, output_logits, label1, N=1
    ):
        if output_logits == None:
            return 1
        label_logits = (
            nn.Softmax(dim=1)(output_logits).detach().clone().to(torch.device("cuda"))
        )
        target_logit = label_logits[0, label1.item()].item()
        label_logits[0, label1.item()] = -float("inf")
        second_logit = label_logits.max().item()
        a = target_logit - second_logit
        n = max(a / target_logit * N, 1)
        print("ratio minus", n)
        # a = new_logit
        # n = max(a / new_logit * N, 1)
        return n

    def prune_mask_unlimited(self, mask_3):
        mask_4 = mask_3.detach().clone()
        with torch.no_grad():
            mask_4 /= mask_4.max()
            mask_4[mask_4.le(0.5)] = 0
            mask_4[mask_4.gt(0.5)] = 1

        mask_4 = mask_4.to(self.device).requires_grad_()
        return mask_4

    def prune_gradient(self, x, mask):
        with torch.no_grad():
            x.grad[~mask] = None

    # def prune_mask(self, gradients):
    #     mask_4 = gradients.detach().clone()
    #     mask_4 = torch.abs(mask_4)
    #     with torch.no_grad():
    #         mask_4 /= mask_4.max()
    #     # prune 3 channels separately
    #     mask_placeholder = torch.zeros(mask_4.shape)
    #     for channel in range(3):
    #         # mask3_single_channel = mask_4.squeeze(0).sum(dim=0)
    #         mask3_single_channel = mask_4.squeeze(0)[channel, :, :]
    #         H, W = mask3_single_channel.shape
    #         num_k = int((1 - self.ratio / 100.0) * mask3_single_channel.numel())
    #         tmp = mask3_single_channel.view(-1)
    #         values, index = tmp.topk(num_k)
    #         two_d_indices = torch.cat(((index // W).unsqueeze(1), (index % W).unsqueeze(1)), dim=1)
    #         mask_placeholder[0, channel, two_d_indices[:, 0], two_d_indices[:, 1]] = 1
    #     with torch.no_grad():
    #         gradients[mask_placeholder.le(0.5)] = 0.0
    #         # mask_4[mask_placeholder.gt(0.5)] = 1.0
    # elif self.args.renorm == 2:
    # mask_4[mask_placeholder.gt(0.5)] = 1
    # mask_4 = mask_4.to(self.device).requires_grad_()
    # return mask_4
