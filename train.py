from omegaconf import DictConfig
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import random
import jax
import jax.numpy as jnp
from flax import nnx
from elements import Space
from tqdm import tqdm
import imageio

from networks.dreamerv3 import Dreamerv3
from networks.agent import ActorCritic
from utils.utils import Logger
from utils.env_wrappers import build_single_env_unified, build_env
from utils.replaybuffer import Replaybuffer, Datasets
from utils.optim import make_simple_opt
import hydra
import swanlab
from omegaconf import OmegaConf

sg = jax.lax.stop_gradient

def seed_np(seed=20010105):
    '''
    设置随机数种子，以确保实验的可重复性。
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

@hydra.main(config_path='./config',config_name='config')
def main(config:DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.training.device)
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    outputs_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    config.agent.feat_dim = config.dreamerv3.rssm.stoch*config.dreamerv3.rssm.stoch+config.dreamerv3.rssm.deter
    log_path = f'{outputs_path}/{config.training.env_name}'
    video_path = os.path.join(log_path,'imagine')
    save_path = os.path.abspath(os.path.join(log_path,'ckpt'))
    os.makedirs(video_path,exist_ok=True)
    os.makedirs(save_path,exist_ok=True)
    
    seed_np(config.training.seed)
    env_type = getattr(config.training, 'env_type', 'atari')
    obs_type = getattr(config.training, 'obs_type', 'image')
    is_atari = (env_type == 'atari')

    dumy_env = build_single_env_unified(config.training.env_name, env_type, obs_type, (64,64), config.training.seed)
    if is_atari:
        action_space = Space(np.uint8, (), 0, dumy_env.action_space.n)
        obs_space = {'image': Space(np.uint8, dumy_env.observation_space.shape, low=0, high=255)}
    else:
        act_dim = dumy_env.action_space.shape[0]
        action_space = Space(np.float32, (act_dim,), -1.0, 1.0)
        if obs_type == 'state':
            state_dim = dumy_env.observation_space.shape[0]
            obs_space = {'state': Space(np.float32, (state_dim,), low=-np.inf, high=np.inf)}
        else:
            obs_space = {'image': Space(np.uint8, dumy_env.observation_space.shape, low=0, high=255)}
    dumy_env.close()

    env = build_env(env_name=config.training.env_name, env_type=env_type, obs_type=obs_type,
                    image_size=(64,64), num_envs=config.training.num_envs, seed=config.training.seed)
    init_key = nnx.Rngs(config.training.seed)

    # Build state encoder config if needed
    state_enc_cfg = None
    state_dec_cfg = None
    if obs_type == 'state':
        state_enc_cfg = getattr(config.dreamerv3, 'state_encoder')
        state_dec_cfg = getattr(config.dreamerv3, 'state_decoder')
        # Auto-detect state_dim from env
        state_dim = obs_space['state'].shape[0]
        if state_enc_cfg.state_dim == 0:
            state_enc_cfg.state_dim = state_dim
        if state_dec_cfg.state_dim == 0:
            state_dec_cfg.state_dim = state_dim
        obs_key = 'state'
    else:
        obs_key = 'image'

    agent = ActorCritic(**config.agent, action_space=action_space, init_key=init_key)
    dreamerv3 = Dreamerv3(enc_cfg=getattr(config,'dreamerv3').encoder,
                          seq_cfg=getattr(config,'dreamerv3').rssm,
                          dec_cfg=getattr(config,'dreamerv3').decoder,
                          action_space=action_space, init_key=init_key,
                          obs_type=obs_type, state_enc_cfg=state_enc_cfg, state_dec_cfg=state_dec_cfg)
    agent_opt = nnx.Optimizer(agent, make_simple_opt(lr=3e-5,grad_norm=100.))
    dreamerv3_opt = nnx.Optimizer(dreamerv3, make_simple_opt(lr=1e-4,grad_norm=1000.))
    if config.training.use_datasets:
        rplb =Datasets(config.training.num_envs,obs_space,action_space,100,int(1E5),2500,config.training.batch_length)
    else:
        rplb = Replaybuffer(obs_space,int(1E5),config.training.num_envs,action_space,1024,config.training.batch_length)
    logger = Logger(log_path)

    exp_name = (f"{config.training.env_name}"
                f"_IL{config.training.imagine_length}"
                f"_g{config.agent.gamma}"
                f"_ent{config.agent.entropy_coef}"
                f"_seed{config.training.seed}")
    swanlab.init(
        project="DreamerV3",
        experiment_name=exp_name,
        config=OmegaConf.to_container(config, resolve=True),
    )
    
    obs, _ = env.reset()
    carry = dreamerv3.init_carry(config.training.num_envs, init_key.noise())
    sum_rewards = np.zeros(config.training.num_envs)
    episodes = 0
    with tqdm(total=config.training.total_steps,desc='Epoch',ncols=120) as pbar:
        for epoch in range(config.training.total_steps):
            if rplb.ready:
                carry,_ = sg(dreamerv3.encode(carry,obs,action,reset))
                feats = jnp.concatenate([carry['deter'],carry['stoch']],axis=-1)
                action, carry['key'] = sg(agent.sample_policy(feats,carry['key']))
                action = np.asarray(action)
            else:
                action = env.action_space.sample()
            next_obs, reward, termination, truncated, info = env.step(action)
            if is_atari:
                reset = np.logical_or(termination, info["life_loss"])
            else:
                reset = termination
            step = {'obs':{obs_key:obs},'action':action,'reward':reward,'termination':reset}
            rplb.add(step)

            done_flags = np.logical_or(termination,truncated)
            if done_flags.any():
                for i in range(config.training.num_envs):
                    if done_flags[i]:
                        episodes += 1
                        env_metrics = {f'Env/episode_return':sum_rewards[i]}
                        if is_atari and 'episode_frame_number' in info:
                            env_metrics[f'Env/episode_length'] = current_info['episode_frame_number'][i]//4
                        elif 'episode_step_count' in info:
                            env_metrics[f'Env/episode_length'] = info['episode_step_count'][i]
                        logger.log_dict(env_metrics)
                        pbar.set_postfix({'episode':episodes,'return':sum_rewards[i]})
                        sum_rewards[i] = 0
            
            sum_rewards += reward
            current_info = info
            obs = next_obs
            
            if epoch*config.training.train_ratio%(config.training.batch_length*config.training.batch_size)==0 and rplb.ready:
                training_metrics = {}
                batch = rplb.sample(config.training.batch_size,config.training.prioritized)
                sample_obs = batch['obs'][obs_key]
                wm_loss, wm_metrics, train_carry = dreamerv3.update(dreamerv3_opt,sample_obs,batch['action'],batch['reward'],
                                                               batch['termination'],carry['key'])
                train_carry, pred_actions, train_feats, pred_rews, pred_ters = \
                    dreamerv3.imagine(train_carry,agent.sample_policy,config.training.imagine_length)
                ac_loss, ac_metrics = agent.update(agent_opt,train_feats,pred_actions,pred_rews,pred_ters)
                training_metrics.update(wm_metrics)
                training_metrics.update(ac_metrics)
                logger.log_dict(training_metrics)
                #imagine video (only for image obs)
                if epoch%5000 == 0 and obs_type == 'image':
                    slide_length = train_feats.shape[0]//16
                    pred_video = sg(dreamerv3.dec(train_feats[::slide_length]))
                    video = np.asarray(pred_video)*255.
                    video = np.clip(video,0,255)
                    video = video.astype(np.uint8)
                    os.makedirs(f'{video_path}/{epoch}',exist_ok=True)
                    swanlab_videos = []
                    for i in range(16):
                        gif_path = f'{video_path}/{epoch}/{i}.gif'
                        imageio.mimsave(gif_path,video[i],fps=20)
                        swanlab_videos.append(swanlab.Image(video[i][0]))
                    swanlab.log({"Imagine/video": swanlab_videos})

                #save models
                if epoch%10000 == 0:
                    dreamerv3.save(os.path.join(save_path,str(epoch)))
                    agent.save(os.path.join(save_path,str(epoch)))

                carry['key'] = train_carry['key'] 

            pbar.update(1)

    swanlab.finish()

if __name__ == '__main__':
    main()