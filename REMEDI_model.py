import argparse
import glob
from datetime import datetime

# Import custom functions for RL environment setup and logging utility
from src.rl_env import psc_ba_env
from src.rl_util import TensorboardCallback

# Import Stable Baselines3 modules for RL (https://stable-baselines3.readthedocs.io/)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3 import PPO, A2C

import wandb
from wandb.integration.sb3 import WandbCallback

def get_args():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, default="PPO")
    parser.add_argument('--train_steps', type=int, default=4e6)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--adaptation_days', type=int, default=240)
    parser.add_argument('--data_ID', type=str, default="median")
    parser.add_argument('--max_ba_flow', type=float, default=3.0)
    parser.add_argument('--gut_deconj_freq_co_multiplier', type=float, default=1.0)
    parser.add_argument('--gut_biotr_freq_CA_multiplier', type=float, default=1.0)
    parser.add_argument('--continue_train_suffix', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':

    wandb.login(key="<WANDB_API_KEY>")

    # Retrieve command-line arguments
    args = get_args()

    # Generate a unique identifier for the current run
    if args.continue_train_suffix==None:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.data_ID}_{args.max_ba_flow}_{args.gut_deconj_freq_co_multiplier}_{args.gut_biotr_freq_CA_multiplier}"
    else:
        # If resuming training from a saved model
        suffix = args.continue_train_suffix
    model_path = f"experiments/{args.algorithm}/logs_{suffix}"

    eval_steps = int(args.train_steps / args.n_envs / 100)
    save_steps = int(args.train_steps / args.n_envs / 5)
    
    # Set up the vectorized RL environment: bile acid metabolism ODEs extended with PSC pathophysiology
    env = make_vec_env(lambda: psc_ba_env(adaptation_duration = args.adaptation_days * 1440,
                                          data_ID = args.data_ID,
                                          max_ba_flow = args.max_ba_flow,
                                          gut_deconj_freq_co_multiplier = args.gut_deconj_freq_co_multiplier,
                                          gut_biotr_freq_CA_multiplier = args.gut_biotr_freq_CA_multiplier),
                       n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    
    # Initialize the RL model
    if args.continue_train_suffix==None:
        
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        # Define the network architecture
        policy_kwargs = dict(net_arch = [100,50,25])

        if args.algorithm == "PPO":
            model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log='experiments/runs_PPO', verbose=1,
                        learning_rate=args.learning_rate, gamma=0.99)        
        elif args.algorithm == "A2C":
            model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log='experiments/runs_A2C', verbose=1,
                        learning_rate=args.learning_rate, gamma=0.99, device="cpu")
        
    else:
        # If resuming training from a saved model, load the model and the environment from the most recent checkpoint
        # (Loading environmnet is needed because observation normalization parameters are saved with the environment)
        files = glob.glob(f"{model_path}/rl_model_*_steps.zip")
        ckpt_steps = [int(f.split("_steps.zip")[0].split("rl_model_")[1]) for f in files]
        up_to_date_model = f"{model_path}/rl_model_{str(max(ckpt_steps))}_steps"
        up_to_date_env = f"{model_path}/rl_model_vecnormalize_{str(max(ckpt_steps))}_steps"
        
        env = VecNormalize.load(f"{up_to_date_env}.pkl", env)
        
        if args.algorithm == "PPO":
            model = PPO.load(f"{up_to_date_model}", env=env)        
        elif args.algorithm == "A2C":
            model = A2C.load(f"{up_to_date_model}", env=env)
        print(f"Loading model from {up_to_date_model}")
            
    # Define callbacks for logging
    checkpoint_callback = CheckpointCallback(save_freq=save_steps, save_path=model_path, save_vecnormalize=True)
    eval_callback = EvalCallback(env, eval_freq=eval_steps, best_model_save_path=model_path, deterministic=False, verbose=0)
    tensorboard_callback = TensorboardCallback(check_freq=eval_steps)

    config = {
        "algorithm": args.algorithm,
        "total_timesteps": args.train_steps,
        "data_ID": args.data_ID,
        "adaptation_days": args.adaptation_days,
        "max_ba_flow": args.max_ba_flow,
        "gut_deconj_freq_co_multiplier": args.gut_deconj_freq_co_multiplier,
        "gut_biotr_freq_CA_multiplier": args.gut_biotr_freq_CA_multiplier,
    }
    run = wandb.init(
        project="remedi_aws",
        name=datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.data_ID}_{args.max_ba_flow}_{args.gut_deconj_freq_co_multiplier}_{args.gut_biotr_freq_CA_multiplier}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    wandb_callback = WandbCallback(gradient_save_freq=eval_steps, model_save_path=f"models/{run.id}", verbose=2)

    callbacks = CallbackList([checkpoint_callback, eval_callback, tensorboard_callback, wandb_callback])
    
    # Train the model
    if args.continue_train_suffix==None:
        model.learn(total_timesteps=args.train_steps, callback=callbacks, tb_log_name=suffix)
    else:
        model.learn(total_timesteps=args.train_steps, callback=callbacks, tb_log_name=suffix, reset_num_timesteps=False)

    # Save the model and the environment after training
    model.save(f"{model_path}/rl_model_after_training")
    env.save(f"{model_path}/rl_model_vecnormalize_after_training.pkl")