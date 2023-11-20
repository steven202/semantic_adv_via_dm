#! /usr/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diff2
trap "echo Exited!; exit;" SIGINT SIGTERM

for diff in 2 ; do
    for mask in 4 3; do 
        CUDA_VISIBLE_DEVICES=3 python main.py --attack  \
                    --config celeba.yml         \
                    --exp "./runs_500_1228/white_"$tune      \
                    --n_test_img 4             \
                    --t_0 500                   \
                    --n_inv_step 40             \
                    --n_test_step 40            \
                    --n_precomp_img 500 --mask $mask --diff $diff --tune 3 --black 0 >> "./runs_500_1228/log_white_"$tune".txt"
    done
done

