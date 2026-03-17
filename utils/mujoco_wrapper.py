import numpy as np
import gymnasium
import jax
import jax.numpy as jnp

try:
    import mujoco
    from mujoco_playground import registry
except ImportError:
    registry = None
    mujoco = None


class MuJoCoPlaygroundEnv(gymnasium.Env):
    """Wraps a MuJoCo Playground env to match Gymnasium step/reset API."""
    metadata = {"render_modes": []}

    def __init__(self, env_name, obs_type='state', image_size=(64, 64), seed=0):
        assert registry is not None, (
            "mujoco_playground is not installed. "
            "Install with: pip install mujoco mujoco-mjx mujoco_playground"
        )
        super().__init__()
        self.env = registry.load(env_name)
        self.obs_type = obs_type
        self.image_size = image_size
        self._rng = jax.random.PRNGKey(seed)
        self._state = None
        self._step_count = 0

        # Action space: continuous, typically [-1, 1]
        act_dim = self.env.action_size
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        if obs_type == 'state':
            # Determine observation dimension
            rng = jax.random.PRNGKey(seed + 9999)
            dummy_state = self.env.reset(rng)
            dummy_obs = self._extract_state_obs(dummy_state)
            obs_dim = dummy_obs.shape[-1]
            self.observation_space = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = gymnasium.spaces.Box(
                low=0, high=255, shape=(*image_size, 3), dtype=np.uint8
            )
            self._renderer = mujoco.Renderer(
                self.env.mj_model, height=image_size[0], width=image_size[1]
            )
            self._mj_data = mujoco.MjData(self.env.mj_model)

    def _extract_state_obs(self, state):
        """Extract flat state observation from environment state."""
        obs = state.obs
        if isinstance(obs, dict):
            parts = [np.asarray(v).flatten() for v in obs.values()]
            return np.concatenate(parts).astype(np.float32)
        return np.asarray(obs, dtype=np.float32).flatten()

    def _render_pixels(self, state):
        """Render pixel observation from MJX state using MuJoCo renderer."""
        qpos = np.asarray(state.data.qpos)
        qvel = np.asarray(state.data.qvel)
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = qvel
        mujoco.mj_forward(self.env.mj_model, self._mj_data)
        self._renderer.update_scene(self._mj_data)
        pixels = self._renderer.render()
        return pixels.astype(np.uint8)

    def _get_obs(self, state):
        if self.obs_type == 'state':
            return self._extract_state_obs(state)
        else:
            return self._render_pixels(state)

    def reset(self, **kwargs):
        self._rng, reset_rng = jax.random.split(self._rng)
        self._state = self.env.reset(reset_rng)
        self._step_count = 0
        obs = self._get_obs(self._state)
        info = {"life_loss": False}
        return obs, info

    def step(self, action):
        action = jnp.array(action, dtype=jnp.float32)
        self._state = self.env.step(self._state, action)
        self._step_count += 1
        obs = self._get_obs(self._state)
        reward = float(self._state.reward)
        terminated = bool(self._state.done)
        truncated = False
        info = {
            "life_loss": False,
            "episode_step_count": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self, '_renderer'):
            self._renderer.close()


def build_mujoco_env(env_name, obs_type='state', image_size=(64, 64), seed=0):
    return MuJoCoPlaygroundEnv(env_name, obs_type, image_size, seed)


def build_mujoco_vec_env(env_name, obs_type='state', image_size=(64, 64), num_envs=1, seed=0):
    def make_env(env_seed):
        return lambda: MuJoCoPlaygroundEnv(env_name, obs_type, image_size, env_seed)
    env_fns = [make_env(seed + i) for i in range(num_envs)]
    return gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)


def build_mujoco_eval_env(env_name, obs_type='state', image_size=(64, 64), seed=0):
    return MuJoCoPlaygroundEnv(env_name, obs_type, image_size, seed)


def build_mujoco_eval_vec_env(env_name, obs_type='state', image_size=(64, 64), num_envs=1, seed=0):
    def make_env(env_seed):
        return lambda: MuJoCoPlaygroundEnv(env_name, obs_type, image_size, env_seed)
    env_fns = [make_env(seed + i) for i in range(num_envs)]
    return gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
