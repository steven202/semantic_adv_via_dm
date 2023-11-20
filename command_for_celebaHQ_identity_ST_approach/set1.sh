#! /usr/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diff2
trap "echo Exited!; exit;" SIGINT SIGTERM
for tune in 0 ; do
    CUDA_VISIBLE_DEVICES=2 python main.py --attack  \
                --config celeba.yml         \
                --exp "./runs_gender_500_0120/black_"$tune      \
                --t_0 500                   \
                --n_inv_step 40             \
                --n_test_step 40            \
                --n_precomp_img 100 --mask 9 --diff 9 --tune $tune --black 1 >> "./runs_gender_500_0120/log_black_"$tune".txt"
done

for tune in 0 ; do
    CUDA_VISIBLE_DEVICES=2 python main.py --attack  \
                --config celeba.yml         \
                --exp "./runs_500_1228/white_"$tune      \
                --n_test_img 4             \
                --t_0 500                   \
                --n_inv_step 40             \
                --n_test_step 40            \
                --n_precomp_img 100 --mask 9 --diff 9 --tune $tune --black 0 >> "./runs_500_1228/log_white_"$tune".txt"
done