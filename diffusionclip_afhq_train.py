import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
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
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.optim as optim
import torch.nn.functional as F
import torchattacks
import lpips


def heat_map(mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    transform = transforms.Grayscale()
    mask = transform(torch.nn.functional.normalize(mask.cpu())).squeeze().squeeze()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = torch.tensor(np.float32(heatmap) / 255).permute(2, 0, 1).unsqueeze(dim=0).cuda()

    return heatmap


class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

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
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == "fixedsmall":
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def attack(self):
        assert self.config.data.dataset == "AFHQ"
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
        for model_path in model_paths:
            if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
                model_i = DDPM(self.config)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                learn_sigma = False
            elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
                if self.config.data.dataset == "AFHQ" and model_path == None:
                    model_path = "dm_afhq_model.pth"
                    with open(os.path.join(self.args.image_folder, f'{"train"}_log.txt'), "a+") as file1:
                        file1.write(f"load afhq model path")
                    print("load afhq model path")
                model_i = i_DDPM(self.config.data.dataset)
                if model_path:
                    ckpt = torch.load(model_path)
                else:
                    ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
                learn_sigma = True
            else:
                print("Not implemented dataset")
                raise ValueError
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
                transforms.RandomHorizontalFlip(),  # data augmentation
                tfs.ToTensor(),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        test_transform = tfs.Compose(
            [
                tfs.ToTensor(),
                transforms.Resize((256, 256)),
                tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        if self.config.data.dataset == "CelebA_HQ" and self.args.face == "identity":
            data_dir = "./CelebA_HQ_facial_identity_dataset"
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
            save_path = "facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"
            desc = "face identity classification"
        elif self.config.data.dataset == "CelebA_HQ" and self.args.face == "gender":
            data_dir = "./CelebA_HQ_face_gender_dataset"
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), test_transform)
            save_path = "facial_gender_classification_transfer_learning_with_ResNet18_resolution_256.pth"
            desc = "face gender classification"
        elif self.config.data.dataset == "AFHQ":
            data_dir = "data/afhq"
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), test_transform)
            save_path = "afhq_resnet18.pth"
            desc = "animal classification"
            save_name = "dm_afhq_model.pth"
        with open(os.path.join(self.args.image_folder, f'{"train"}_log.txt'), "a+") as file1:
            file1.write(desc + "\n")
        loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=2, num_workers=self.config.data.num_workers)
        loader = loader_dic["train"]

        net = torchvision.models.resnet18(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, len(train_dataset.classes))  # multi-class classification (num_of_class == 307)
        net = net.to(self.device)
        net.load_state_dict(torch.load(save_path))
        net.eval()

        # ----------- Optimizer and Scheduler -----------#
        model = models[1]
        model.module.load_state_dict(ckpt)
        model.to(self.device)
        # model = torch.nn.DataParallel(model)

        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optim_ft, step_size=1, gamma=self.args.sch_gamma)
        init_sch_ckpt = scheduler_ft.state_dict()
        # ----------- Loss -----------#
        print("Loading losses")
        loss_fn_alex = lpips.LPIPS(net="alex").to(self.device)
        id_loss_func = id_loss.IDLoss().to(self.device).eval()

        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        finetune_counter = 0
        for index, mode in enumerate(tqdm((["train"] * 500000))):
            print("INDEX:", index)
            print()
            img_lat_pairs = []
            count = 0
            total = 0
            for step, (images, labels) in enumerate(loader):
                img1 = images[0].unsqueeze(dim=0)
                x0 = img1.to(self.config.device)
                x_0 = x0.detach().clone()
                label1 = labels[0].detach().clone().cuda()
                x_0_ = x_0.detach().clone()
                # 1. generate two latent spaces
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x_0_ = denoising_step(
                                x_0_,
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

                            progress_bar.update(1)

                    x_lat_0 = x_0_.detach().clone()

                    # x = x_lat.detach().clone().cuda() # original image
                    img_lat_pair = [x_0.detach().clone(), None, x_lat_0.detach().clone()]
                    img_lat_pairs_dic = [x_0.detach().clone(), None, x_lat_0.detach().clone()]
                # ----------- Finetune Diffusion Models -----------#
                print("Start finetuning")
                print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
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

                # model.module.load_state_dict(ckpt)
                # optim_ft.load_state_dict(init_opt_ckpt)
                # scheduler_ft.load_state_dict(init_sch_ckpt)
                # ----------- Train -----------#
                for it_out in range(self.args.n_iter):
                    exp_id = os.path.split(self.args.exp)[-1]
                    # save_name = f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                    x_0_train = img_lat_pair[0]
                    x_lat_train = img_lat_pair[2]

                    for step2 in range(1):
                        model.train()
                        time_in_start = time.time()

                        optim_ft.zero_grad()
                        x_train = x_lat_train.detach().clone()

                        with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                            for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                t = (torch.ones(n) * i).to(self.device)
                                t_next = (torch.ones(n) * j).to(self.device)

                                x_train = denoising_step(
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

                                progress_bar.update(1)

                                # loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                                # loss_clip = -torch.log(loss_clip)
                        # loss_clip = loss_fn_alex(x_0_train,x_train)
                        # loss_id = torch.mean(id_loss_func(x_0_train, x_train))
                        loss_l1 = nn.L1Loss()(x_0_train, x_train)
                        loss = self.args.l1_loss_w * loss_l1
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optim_ft.step()
                        finetune_counter += 1
                        print(f"CLIP index {index} step {step}-{it_out}: loss_l1: {loss_l1.item():.3f}")
                        if step == 0 or (step + 1) % 10 == 0:
                            if self.args.save_train_image:
                                tvu.save_image(
                                    (x_0_train + 1) * 0.5,
                                    os.path.join(self.args.image_folder, f"train_{step}_2_orig_{self.args.n_train_step}.png"),
                                )
                            if self.args.save_train_image:
                                tvu.save_image(
                                    (x_train + 1) * 0.5,
                                    os.path.join(self.args.image_folder, f"train_{step}_2_gene_{self.args.n_train_step}.png"),
                                )
                        time_in_end = time.time()
                        print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")

                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), save_name)
                        models[0].module.load_state_dict(model.module.state_dict())
                    else:
                        torch.save(model.state_dict(), save_name)
                        models[0].load_state_dict(model.state_dict())
                    print(f"Model {save_name} is saved.")
                    scheduler_ft.step()

                    # print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
                    # if self.args.n_test_step != 0:
                    #     seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                    #     seq_test = [int(s) for s in list(seq_test)]
                    #     print('Uniform skip type')
                    # else:
                    #     seq_test = list(range(self.args.t_0))
                    #     print('No skip')
                    # seq_test_next = [-1] + list(seq_test[:-1])
                    # print("Start evaluation")
                    # eval_modes = ['test']
                    # for mode in eval_modes:
                    #     x_0_eval = img_lat_pairs_dic[0]
                    #     x_lat_eval = img_lat_pairs_dic[2]
                    #     for step in [0]:
                    #         with torch.no_grad():
                    #             x_eval = x_lat_eval.detach().clone()
                    #             with tqdm(total=len(seq_test), desc=f"Eval iteration") as progress_bar:
                    #                 for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    #                     t = (torch.ones(n) * i).to(self.device)
                    #                     t_next = (torch.ones(n) * j).to(self.device)

                    #                     x_eval = denoising_step(x_eval, t=t, t_next=t_next, models=models,
                    #                                             logvars=self.logvar,
                    #                                             sampling_type=self.args.sample_type,
                    #                                             b=self.betas,
                    #                                             eta=self.args.eta,
                    #                                             learn_sigma=learn_sigma,
                    #                                             ratio=self.args.model_ratio,
                    #                                             hybrid=self.args.hybrid_noise,
                    #                                             hybrid_config=HYBRID_CONFIG)

                    #                     progress_bar.update(1)

                    #             print(f"Eval {step}")
                    #             tvu.save_image((x_eval + 1) * 0.5,
                    #                                 os.path.join(self.args.image_folder,
                    #                                                 f'{mode}_{step}_2_clip_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png'))

                # if self.args.smooth == 1:
                #     x_eval, pred3, new_logit = self.smooth_process(models, learn_sigma, net, seq_inv, seq_inv_next, n, mode, img_lat_pairs, step, x0.detach().clone().cuda(), x_eval.detach().clone().cuda(), label1)
                # tvu.save_image((x_eval + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_fti_{fixed_finetunes}_only_reconstruction.png'))
                # tvu.save_image((x_0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_fti_{fixed_finetunes}_only_original.png'))
                # grid_images = torchvision.utils.make_grid(torch.cat(((x0 + 1) * 0.5,(x1 + 1) * 0.5,heat_map(mask=(x_eval-x_0),use_rgb=True),(x_eval + 1) * 0.5),dim=0),nrow=8)
                # tvu.save_image(grid_images, os.path.join(self.args.image_folder, f'{mode}_{step}_{"true" if pred1!=pred3 else "false"}_fti_{fixed_finetunes}_lbl_{label1.item()}_pred1_{pred1}_pred2_{pred3}_1_rec.png'))
                # count += 1 if pred1!=pred3 else 0
                # total += 1
                # print(f'{mode}_{step}_1_asr_{count*1.0/total}_count_{count}_total_{total}\n')
                # with open(os.path.join(self.args.image_folder, f'{mode}_log.txt'), "a+") as file1:
                #     file1.write(f'Per image: {mode}_{step}_fti_{fixed_finetunes}_{"true" if pred1!=pred3 else "false"}_logit1_{"{:.2f}".format(original_logit)}_logit2_{"{:.2f}".format(new_logit)}_lbl_{label1.item()}_pred1_{pred1}_pred2_{pred3}_1_rec\n')
                #     file1.write(f'Total count: {mode}_step_{step}_ratio_asr_{count*1.0/total}_count_{count}_total_{total}\n')
                # if total >= self.args.n_precomp_img:
                #     break

    def mask_method(
        self,
        models,
        learn_sigma,
        net,
        eps,
        seq_inv,
        seq_inv_next,
        n,
        mode,
        img_lat_pairs,
        step,
        x0,
        label1,
        x_lat,
        x_lat__,
        mask3_channels,
        ratio,
        mask_4,
    ):
        #######################################################
        # threshold_3 = torch.quantile(mask_4.flatten(),ratio/100.)
        if True:
            if True:
                mask3_single_channel = mask3_channels.sum(dim=0)
                H, W = mask3_single_channel.shape
                num_k = int((1 - ratio / 100.0) * mask3_single_channel.numel())
                tmp = mask3_single_channel.view(-1)
                values, index = tmp.topk(num_k)
                two_d_indices = torch.cat(((index // W).unsqueeze(1), (index % W).unsqueeze(1)), dim=1)
                mask_placeholder = torch.zeros(mask_4.shape)
                mask_placeholder[0, :, two_d_indices[:, 0], two_d_indices[:, 1]] = 1
            elif True:
                mask_placeholder = torch.zeros(mask_4.shape)
                for channel in range(3):
                    mask3_single_channel = mask3_channels.sum(dim=0)
                    # mask3_single_channel = mask3_channels[channel,:,:]
                    H, W = mask3_single_channel.shape
                    num_k = int((1 - ratio / 100.0) * mask3_single_channel.numel())
                    tmp = mask3_single_channel.view(-1)
                    values, index = tmp.topk(num_k)
                    two_d_indices = torch.cat(((index // W).unsqueeze(1), (index % W).unsqueeze(1)), dim=1)
                    mask_placeholder[0, channel, two_d_indices[:, 0], two_d_indices[:, 1]] = 1
                    # v, i = torch.topk(mask3_single_channel.flatten(), num_k)
                    # indices = np.array(np.unravel_index(i.cpu().numpy(), mask3_single_channel.shape)).T
                    # values = mask3_single_channel[indices]
                    # mask_placeholder[0,channel][two_d_indices]=1
            mask_4[mask_placeholder.le(0.5)] = 0
            # mask_4[mask_placeholder.gt(0.5)] = 1
        else:
            threshold_3 = torch.quantile(mask_4.flatten(), ratio / 100.0)
            mask_4[mask_4 < threshold_3] = 0
        if self.args.renorm == 0:
            mask_4 /= mask_4.max()
        else:
            mask_4 = mask_4.renorm(p=2, dim=0, maxnorm=eps)
        mask_4 = mask_4.cuda()
        with torch.no_grad():
            x_generate = x_lat__ * mask_4 + x_lat * (1 - mask_4)

            with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x_generate = denoising_step(
                        x_generate,
                        t=t,
                        t_next=t_next,
                        models=models,
                        logvars=self.logvar,
                        sampling_type=self.args.sample_type,
                        b=self.betas,
                        eta=self.args.eta,
                        learn_sigma=learn_sigma,
                        ratio=0,
                    )

                    progress_bar.update(1)
            img_lat_pairs.append([x0, x_generate.detach().clone(), x_lat.detach().clone()])
        with torch.no_grad():
            output3 = net(x_generate)
            pred3 = output3.max(dim=1)[1].item()
            new_logit = nn.Softmax(dim=1)(output3)[0, label1.item()].item()
        #######################################################################
        return mask_4, x_generate, pred3

    def mask6_get_grad(self, learn_sigma, model, loss_fn_alex, id_loss_func, n, x_lat_train, x_0_train_):
        model.train()
        time_in_start = time.time()

        x_train = x_lat_train.detach().clone()
        x_0_train = x_0_train_.detach().clone()
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
        with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
            for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                t = (torch.ones(n) * i).to(self.device)
                t_next = (torch.ones(n) * j).to(self.device)

                x_train = denoising_step(
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

                progress_bar.update(1)

        loss_clip = loss_fn_alex(x_0_train, x_train)
        loss_id = torch.mean(id_loss_func(x_0_train, x_train))
        loss_l1 = nn.L1Loss()(x_0_train, x_train)
        loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1
        grad = torch.autograd.grad(loss, x_train)[0]
        time_in_end = time.time()
        print(f"get grad for 1 image takes {time_in_end - time_in_start:.4f}s")
        return grad

    def mask6_helper(self, models, learn_sigma, seq_inv, seq_inv_next, n, mode, step, x, x__):
        with torch.no_grad():
            with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(
                        x,
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

                    progress_bar.update(1)

            x_lat = x.detach().clone()
            with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x__ = denoising_step(
                        x__,
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

                    progress_bar.update(1)

            x_lat__ = x__.detach().clone()
        return x, x__, x_lat, x_lat__

    def smooth_process(self, models, learn_sigma, net, seq_inv, seq_inv_next, n, mode, img_lat_pairs, step, x0, x, label1):
        # if self.args.eta<0.00000001:
        #     self.args.eta = 0.05
        with torch.no_grad():
            with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_prev = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(
                        x,
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

                    progress_bar.update(1)

            x_lat = x.detach().clone()

            x = x_lat

            with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)

                    x = denoising_step(
                        x,
                        t=t,
                        t_next=t_next,
                        models=models,
                        logvars=self.logvar,
                        sampling_type=self.args.sample_type,
                        b=self.betas,
                        eta=self.args.eta,
                        learn_sigma=learn_sigma,
                        ratio=0,
                    )

                    progress_bar.update(1)

            img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
        with torch.no_grad():
            output3 = net(x)
            pred3 = output3.max(dim=1)[1].item()
            new_logit = nn.Softmax(dim=1)(output3)[0, label1.item()].item()
        return x, pred3, new_logit


def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float() / float(true.size(0))
    return accuracy.item()
