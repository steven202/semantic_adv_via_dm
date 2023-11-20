CUDA_VISIBLE_DEVICES=2 python main_afhq_train.py --attack  \
            --config afhq.yml         \
            --exp ./runs_train_afhq_0122      \
            --n_test_img 4             \
            --t_0 500                   \
            --n_inv_step 40             \
            --n_test_step 40      >> log_train_afhq_0122.txt