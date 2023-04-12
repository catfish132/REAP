# train with regular learning
  CUDA_VISIBLE_DEVICES=0 python main_train.py --cfg path2cfg
  # example: CUDA_VISIBLE_DEVICES=0 python main_train.py --cfg cfg/MIM/MIM_ch13.yml

# train with reinforcement learning
  CUDA_VISIBLE_DEVICES=0 python main_train_rl.py --cfg path2cfg
  # example: CUDA_VISIBLE_DEVICES=0 python main_train_rl.py --cfg cfg_rl/MIM/RL_MIM_ch13.yml

# test with trained model
  CUDA_VISIBLE_DEVICES=0 python main_test.py --cfg path2cfg
  # example: CUDA_VISIBLE_DEVICES=0 python main_test.py --cfg cfg/TEST/MIM/TEST_RL_MIM_xrv18.yml
  CUDA_VISIBLE_DEVICES=7 python main_test.py --cfg cfg/TEST/MIM/TEST_RL_MIM_xrv18.yml