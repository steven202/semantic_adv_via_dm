#! /usr/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate diff2
trap "echo Exited!; exit;" SIGINT SIGTERM
for tune in 2 1 0 ; do
    CUDA_VISIBLE_DEVICES=2 python main.py --attack  \
                --config afhq.yml         \
                --exp "./runs_afhq_500_0120/black_"$tune      \
                --t_0 500                   \
                --n_inv_step 40             \
                --n_test_step 40            \
                --n_precomp_img 500 --face gender --mask 9 --diff 9 --tune $tune --black 1 >> "./runs_afhq_500_0120/log_black_"$tune".txt"
done

# for diff in 3 2 1 ; do
#     for mask in 4 ; do 
#         CUDA_VISIBLE_DEVICES=3 python main.py --attack  \
#                     --config afhq.yml         \
#                     --exp "./runs_afhq_500_0120/white_"$tune      \
#                     --t_0 500                   \
#                     --n_inv_step 40             \
#                     --n_test_step 40            \
#                     --n_precomp_img 500 --face gender --mask $mask --diff $diff --tune 3 --black 0 >> "./runs_afhq_500_0120/log_white_"$tune".txt"
#     done
# done


