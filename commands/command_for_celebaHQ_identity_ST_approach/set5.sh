#! /usr/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diff2
trap "echo Exited!; exit;" SIGINT SIGTERM
for tune in 2 1 0 ; do
    CUDA_VISIBLE_DEVICES=2 python main.py --attack  \
                --config celeba.yml         \
                --exp "./runs_500_0427/white_"$tune      \
                --n_test_img 500             \
                --t_0 500                   \
                --n_inv_step 40             \
                --n_test_step 40            \
                --n_precomp_img 500 --mask 9 --diff 9 --tune $tune --black 0 >> "./runs_500_0427/log_white_"$tune".txt"
done