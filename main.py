import argparse
import datetime
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from diffusionclip import DiffusionCLIP

device = torch.device("cuda")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    # Mode
    parser.add_argument("--clip_finetune", action="store_true")
    parser.add_argument("--clip_latent_optim", action="store_true")
    parser.add_argument("--edit_images_from_dataset", action="store_true")
    parser.add_argument("--edit_one_image", action="store_true")
    parser.add_argument("--unseen2unseen", action="store_true")
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--clip_finetune_eff", action="store_true")
    parser.add_argument("--edit_one_image_eff", action="store_true")

    # Default
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--exp", type=str, default="./runs/", help="Path for saving running related data.")
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", type=int, default=1, help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument("--align_face", type=int, default=1, help="align face or not")

    # Text
    parser.add_argument("--edit_attr", type=str, default=None, help="Attribute to edit defiend in ./utils/text_dic.py")
    parser.add_argument("--src_txts", type=str, action="append", help="Source text e.g. Face")
    parser.add_argument("--trg_txts", type=str, action="append", help="Target text e.g. Angry Face")
    parser.add_argument("--target_class_num", type=str, default=None)

    # Sampling
    parser.add_argument("--t_0", type=int, default=400, help="Return step in [0, 1000)")
    parser.add_argument("--n_inv_step", type=int, default=40, help="# of steps during generative pross for inversion")
    parser.add_argument("--n_train_step", type=int, default=6, help="# of steps during generative pross for train")
    parser.add_argument("--n_test_step", type=int, default=40, help="# of steps during generative pross for test")
    parser.add_argument(
        "--sample_type", type=str, default="ddim", help="ddpm for Markovian sampling, ddim for non-Markovian sampling"
    )
    parser.add_argument("--eta", type=float, default=0.0, help="Controls of varaince of the generative process")

    # Train & Test
    parser.add_argument("--do_train", type=int, default=1, help="Whether to train or not during CLIP finetuning")
    parser.add_argument("--do_test", type=int, default=1, help="Whether to test or not during CLIP finetuning")
    parser.add_argument(
        "--save_train_image", type=int, default=1, help="Wheter to save training results during CLIP fineuning"
    )
    parser.add_argument("--bs_train", type=int, default=16, help="Training batch size during CLIP fineuning")
    parser.add_argument("--bs_test", type=int, default=1, help="Test batch size during CLIP fineuning")
    parser.add_argument("--n_precomp_img", type=int, default=100, help="# of images to precompute latents")
    parser.add_argument("--n_train_img", type=int, default=50, help="# of training images")
    parser.add_argument("--n_test_img", type=int, default=10, help="# of test images")
    parser.add_argument("--model_path", type=str, default=None, help="Test model path")
    parser.add_argument("--img_path", type=str, default=None, help="Image path to test")
    parser.add_argument(
        "--deterministic_inv", type=int, default=1, help="Whether to use deterministic inversion during inference"
    )
    parser.add_argument(
        "--hybrid_noise", type=int, default=0, help="Whether to change multiple attributes by mixing multiple models"
    )
    parser.add_argument(
        "--model_ratio", type=float, default=1, help="Degree of change, noise ratio from original and finetuned model."
    )
    # parser.add_argument("--ratio", "-r", type=int, default=95, help="ratio for torch.quantile.")

    # Loss & Optimization

    parser.add_argument("--clip_model_name", type=str, default="ViT-B/16", help="ViT-B/16, ViT-B/32, RN50x16 etc")
    parser.add_argument("--lr_clip_finetune", type=float, default=2e-6, help="Initial learning rate for finetuning")
    parser.add_argument("--lr_clip_lat_opt", type=float, default=2e-2, help="Initial learning rate for latent optim")
    parser.add_argument(
        "--n_iter", type=int, default=1, help="# of iterations of a generative process with `n_train_img` images"
    )
    parser.add_argument("--scheduler", type=int, default=1, help="Whether to increase the learning rate")
    parser.add_argument("--sch_gamma", type=float, default=1.3, help="Scheduler gamma")

    parser.add_argument("--mask", type=int, default=3, help="what mask to use.")
    parser.add_argument("--diff", type=int, default=4, help="what mask to use.")
    parser.add_argument("--resume_step", type=int, default=0, help="what mask to use.")
    # parser.add_argument("--renorm", type=int, default=0, help="what mask to use.")
    # parser.add_argument("--smooth", type=int, default=0, help="what mask to use.")
    parser.add_argument(
        "--face", type=str, default="identity", choices=["identity", "gender"], help="what mask to use."
    )
    # parser.add_argument("--start", type=int, default=0, help="what mask to use.")
    parser.add_argument("--box", type=str, choices=["white", "black"], help="white or black box attack")
    parser.add_argument(
        "--tune",
        default=1,
        type=int,
        help="finetune latent, model parameter, or latent and model parameter together",
    )
    parser.add_argument(
        "--black",
        type=int,
        default=0,
        choices=[0, 1],
        help="blackbox or not",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        choices=[0, 1],
        help="train adversarial or not",
    )
    parser.add_argument("--clip_loss_w", type=int, default=3, help="Weights of CLIP loss")
    parser.add_argument("--l1_loss_w", type=float, default=1, help="Weights of L1 loss")
    parser.add_argument("--id_loss_w", type=float, default=1, help="Weights of ID loss")
    parser.add_argument("--beta", type=float, default=3, help="Weights of ID loss")
    args = parser.parse_args()
    today = datetime.datetime.now().strftime("%m%d%Y%H%M%S.%f")
    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    box = "black" if args.black == 1 else "white"
    # args.exp = (
    #     args.exp
    #     + f"_box_{box}_tune_{args.tune}_mask{args.mask}_diff{args.diff}_eta{args.eta}_ED_{new_config.data.category}_t{args.t_0}_ninv{args.n_inv_step}_ngen{args.n_train_step}"
    # )
    args.exp = args.exp + f"_box_{box}_tune_{args.tune}_mask{args.mask}_diff{args.diff}_{args.face}"
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(args.exp, exist_ok=True)
    args.image_folder = os.path.join(args.exp, "image_samples")
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            # shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder, exist_ok=True)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    runner = DiffusionCLIP(args, config)
    try:
        if args.attack:
            runner.attack()
        else:
            print("Choose one mode!")
            raise ValueError
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
