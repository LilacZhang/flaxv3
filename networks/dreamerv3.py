import pathlib
import sys
import math
from dataclasses import dataclass
from typing import List,Dict,Callable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import jax 
import jax.numpy as jnp
from flax import nnx
from elements import Space
from einops import reduce,rearrange
import optax
import orbax.checkpoint as ocp

from .rssm import Encoder,RSSM,Decoder,StateEncoder
from .net import MLPHead, Harmonizer
from utils.functional import SymLogTwoHot, Binary, ImageMSE, StateMSE
from utils.optim import make_opt, make_simple_opt
from .agent import ActorCritic

sg = jax.lax.stop_gradient

@dataclass
class EncoderConfig:
    image_shape:tuple=(64,64,3)
    depths:int = 32
    mults:tuple = (1,2,4,8)
    act:str = 'silu'
    norm:str = 'BatchNorm'

@dataclass
class RSSMConfig:
    hidden:int = 512
    layers:int = 1
    deter:int = 512
    group:int = 8
    stoch:int = 32
    classes:int = 32
    act:str = 'silu'
    norm:str = 'LayerNorm'

@dataclass
class DecoderConfig:
    channels:int = 3
    depths:int = 32
    mults:tuple = (1,2,4,8)
    act:str = 'silu'
    norm:str = 'BatchNorm'

@dataclass
class StateEncoderConfig:
    state_dim:int = 0  # auto-detected from env, 0 = auto
    hidden:int = 512
    layers:int = 3
    act:str = 'silu'
    norm:str = 'LayerNorm'

@dataclass
class StateDecoderConfig:
    state_dim:int = 0
    hidden:int = 512
    layers:int = 3
    act:str = 'silu'
    norm:str = 'LayerNorm'


class StateDecoderModule(nnx.Module):
    """MLP decoder for state vector observations (replaces CNN Decoder for state obs)."""
    def __init__(self, feats_dim:int, state_dim:int, hidden:int, layers:int, act:str, norm:str, init_key:nnx.Rngs):
        super().__init__()
        features = [feats_dim] + [hidden]*(layers-1) + [state_dim]
        linear_modules = []
        norm_modules = []
        for in_f, out_f in zip(features[:-2], features[1:-1]):
            linear_modules.append(nnx.Linear(in_f, out_f, rngs=init_key))
            norm_modules.append(getattr(nnx, norm)(out_f, rngs=init_key))
        self.linear_modules = linear_modules
        self.norm_modules = norm_modules
        self.out_proj = nnx.Linear(features[-2], state_dim, rngs=init_key)
        self.act = getattr(nnx, act)

    def __call__(self, x:jax.Array):
        for linear, norm in zip(self.linear_modules, self.norm_modules):
            x = self.act(norm(linear(x)))
        return self.out_proj(x)

class Dreamerv3(nnx.Module):
    def __init__(self, enc_cfg:EncoderConfig, seq_cfg:RSSMConfig, dec_cfg:DecoderConfig,
                 action_space:Space, init_key:nnx.Rngs,
                 obs_type:str='image', state_enc_cfg=None, state_dec_cfg=None):
        super().__init__()
        self.obs_type = obs_type
        feats_dim = seq_cfg.deter + seq_cfg.stoch * seq_cfg.classes

        if obs_type == 'state':
            assert state_enc_cfg is not None, "state_enc_cfg required for obs_type='state'"
            # Compute token_dim to match what CNN encoder would produce
            img_shape = list(enc_cfg.image_shape)
            for _ in enc_cfg.mults:
                img_shape[0] //= 2
                img_shape[1] //= 2
            img_shape[2] = enc_cfg.depths * enc_cfg.mults[-1]
            token_dim = math.prod(img_shape)
            self.enc = StateEncoder(
                state_dim=state_enc_cfg.state_dim, hidden=state_enc_cfg.hidden,
                layers=state_enc_cfg.layers, token_dim=token_dim,
                act=state_enc_cfg.act, norm=state_enc_cfg.norm, init_key=init_key
            )
            state_dim = state_enc_cfg.state_dim
            sd_cfg = state_dec_cfg if state_dec_cfg is not None else state_enc_cfg
            self.dec = StateDecoderModule(
                feats_dim=feats_dim, state_dim=state_dim,
                hidden=sd_cfg.hidden, layers=sd_cfg.layers,
                act=sd_cfg.act, norm=sd_cfg.norm, init_key=init_key
            )
        else:
            self.enc = Encoder(image_shape=enc_cfg.image_shape, depths=enc_cfg.depths, mults=enc_cfg.mults,
                               act=enc_cfg.act, norm=enc_cfg.norm, init_key=init_key)
            self.dec = Decoder(feats_dim=feats_dim, final_shape=self.enc.final_shape,
                               channels=dec_cfg.channels, depths=dec_cfg.depths, mults=dec_cfg.mults,
                               act=dec_cfg.act, norm=dec_cfg.norm, init_key=init_key)

        self.seq = RSSM(final_shape=self.enc.final_shape, hidden=seq_cfg.hidden, layers=seq_cfg.layers,
                        deter=seq_cfg.deter, group=seq_cfg.group, stoch=seq_cfg.stoch, classes=seq_cfg.classes,
                        action_space=action_space, act=seq_cfg.act, norm=seq_cfg.norm, init_key=init_key)
        self.rew = MLPHead(in_features=feats_dim, out_features=255,
                           hidden_features=seq_cfg.hidden, layers=3, act=seq_cfg.act, norm=seq_cfg.norm,
                           init_key=init_key)
        self.ter = MLPHead(in_features=feats_dim, out_features=1,
                           hidden_features=seq_cfg.hidden, layers=3, act=seq_cfg.act, norm=seq_cfg.norm,
                           init_key=init_key)
        self.rew_out = SymLogTwoHot(bins=255, min_val=-20., max_val=20.)
    
    def init_carry(self, batch_size:int, key:jax.random.PRNGKey):
        carry = self.seq.init_recurrent(batch_size, key)
        return carry
      
    def encode(self,carry:Dict,obs:jax.Array,action:jax.Array,reset:jax.Array):
        tokens = self.enc(obs)
        if self.obs_type != 'state':
            tokens = rearrange(tokens, '... H W C -> ... (H W C)')
        if tokens.ndim == 2:
            carry, feats = self.seq._observe(carry, tokens, action, reset)
        else:
            carry, feats = self.seq.observe(carry, tokens, action, reset)
        return carry, feats
    
    def decode(self,deter:jax.Array, stoch:jax.Array):
        feats = jnp.concatenate([deter, stoch], axis=-1)
        rec_images = self.dec(feats)
        return rec_images
    
    def imagine(self, carry:Dict, policy:Callable, imagine_length:int):
        carry,feats = self.seq.imagine(carry,policy,imagine_length)
        img_actions = feats['action']
        feats = jnp.concatenate([feats['deter'],feats['stoch']],axis=-1)
        rew_weights = jax.nn.softmax(self.rew(feats[:,1:])[0],axis=-1) #这里还真不好说
        pred_rews = self.rew_out.decode(rew_weights)
        ter_logits = self.ter(feats[:,1:])[0]
        pred_ters = Binary(ter_logits).prob(1)
        return sg(carry), sg(img_actions), sg(feats), sg(pred_rews), sg(pred_ters)
    
    @staticmethod
    def compute_loss(model:nnx.Module, obs:jax.Array, actions:jax.Array, rewards:jax.Array, terminations:jax.Array, key:jax.random.PRNGKey):
        (actions,rewards,terminations) = jax.device_put((actions,rewards,terminations))
        batch_size = obs.shape[0]
        carry = model.seq.init_recurrent(batch_size, key)
        tokens = model.enc(obs)
        if model.obs_type != 'state':
            tokens = rearrange(tokens, '... H W C -> ... (H W C)')
        carry, feats, seq_losses = model.seq.compute_loss(carry, tokens, actions, terminations)
        pred_feats = jnp.concatenate([feats['deter'],feats['stoch']],axis=-1)
        rewards_logits = model.rew(pred_feats)[0]
        rew_loss = model.rew_out.compute_loss(rewards[:,1:], rewards_logits[:,1:])
        terminations_logits = model.ter(pred_feats)[0]
        terminations_loss = Binary(terminations_logits[:,1:]).loss(terminations[:,1:])
        terminations_loss = jnp.mean(terminations_loss)
        rec = model.dec(pred_feats)
        if model.obs_type == 'state':
            rec_loss = StateMSE(rec, obs)
        else:
            rec_loss = ImageMSE(rec, obs)

        total_loss = rec_loss + rew_loss + terminations_loss + 0.5*seq_losses['dyn_loss'] + 0.1*seq_losses['rep_loss']
        loss_name = 'WorldModel/State_Loss' if model.obs_type == 'state' else 'WorldModel/Image_Loss'
        metrics = {loss_name:rec_loss,
                   'WorldModel/Reward_Loss':rew_loss,
                   'WorldModel/Termination_Loss':terminations_loss,
                   'WorldModel/Dynamic_Loss':seq_losses['dyn_loss'],
                   'WorldModel/Representaion_Loss':seq_losses['rep_loss'],
                   'WorldModel/Total_Loss':total_loss}

        stoch = rearrange(feats['stoch'],'B L D -> (B L) D')
        deter = rearrange(feats['deter'],'B L D -> (B L) D')
        entries = dict(key=carry['key'],stoch=stoch,deter=deter)

        return total_loss, (sg(entries), metrics)
    
    @nnx.jit
    def update(self, optimizer:nnx.Optimizer, obs:jax.Array, actions:jax.Array, rewards:jax.Array, terminations:jax.Array, key:jax.random.PRNGKey):
        (wm_loss, (carry, metrics)), grad = nnx.value_and_grad(self.compute_loss,has_aux=True)(self,obs,actions,rewards,terminations,key)
        optimizer.update(grad)
        global_norm = optax.global_norm(grad)
        metrics['WorldModel/grad_norm'] = global_norm
        return sg(wm_loss), sg(metrics), carry
    
    def save(self, save_path:str):
        _, state = nnx.split(self)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path,'worldmodel')
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(save_path, state)
        
if __name__ == '__main__':
    # import numpy as np
    # from tqdm import tqdm
    # init_key = nnx.Rngs(0)
    # action_space = Space(np.uint8,(),0,18)
    # enc_cfg = EncoderConfig()
    # dec_cfg = DecoderConfig()
    # seq_cfg = RSSMConfig()
    # dreamerv3 = Dreamerv3(enc_cfg,seq_cfg,dec_cfg,action_space,init_key)
    # agent = ActorCritic(1536,512,2,action_space,0.97,0.5,1e-4,'silu','LayerNorm',init_key)
    
    # images = jax.random.randint(init_key.noise(),(16,64,64,64,3),0,255,jnp.uint8)
    # actions = jax.random.randint(init_key.noise(), (16,64), 0, 12)
    # terminations = jax.random.normal(init_key.noise(),(16,64)) > -0.2
    # rewards = jax.random.normal(init_key.noise(),(16,64))
    
    # with tqdm(total=1000,desc='Epoch') as pbar:
    #     for epoch in range(1000):
    #         wm_loss, metrics, carry = dreamerv3.update(images, actions, rewards, terminations, init_key.noise())
    #         carry, pred_actions, feats, pred_rews, pred_ters = dreamerv3.imagine(carry,agent.sample_policy,16)
    #         ac_loss, ac_metrics = agent.update(feats, pred_actions, pred_rews, pred_ters)
    #         metrics.update(ac_metrics)
    #         pbar.set_postfix({'Epoch':epoch+1,'wm_loss':wm_loss,'ac_loss':ac_loss})
    #         pbar.update(1)
    pass