# Flaxv3

Flaxv3 is a simplified version of DreamerV3, composed of Flax and JAX instead of Ninjax, which was used by Hafner in DreamerV3.

## Setup

To set up your Python environment, please install the required dependencies listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## Configuration

To modify the hyperparameters for the algorithm, refer to the [config file](./config/config.yaml).

## Training

### Atari

```bash
python train.py training.env_name=Boxing training.seed=1 training.device=0
```

支持的 Atari 游戏：`Boxing`, `Breakout`, `Pong` 等 Gymnasium Atari 环境。

### MuJoCo Playground

MuJoCo Playground 提供基于 MJX（JAX 后端）的连续控制环境，包括经典控制、locomotion 和机械臂操作任务。

**图像观测（image）训练：**
```bash
python train.py training.env_name=CartpoleBalance training.env_type=mujoco_playground training.obs_type=image
```

**状态观测（state）训练（更快，不需要渲染图像）：**
```bash
python train.py training.env_name=CartpoleBalance training.env_type=mujoco_playground training.obs_type=state
```

**推荐环境（按复杂度排序）：**

| 环境名 | 类型 | 说明 |
|--------|------|------|
| `PointMass` | 经典控制 | 最简单，2D 点质量移动到目标 |
| `PendulumSwingup` | 经典控制 | 单自由度摆杆摆起 |
| `CartpoleBalance` | 经典控制 | 倒立摆平衡 |
| `CartpoleSwingup` | 经典控制 | 倒立摆摆起+平衡 |
| `ReacherEasy` | 经典控制 | 二连杆到达目标 |
| `CheetahRun` | Locomotion | 半猎豹奔跑 |
| `WalkerWalk` | Locomotion | 双足行走 |
| `PandaPickCube` | 操作 | Panda 机械臂抓取方块 |
| `PandaRobotiqPushCube` | 操作 | 机械臂推方块 |

完整环境列表见 [MuJoCo Playground 文档](https://github.com/google-deepmind/mujoco_playground)。

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `training.env_name` | 环境名称 | `Boxing` |
| `training.env_type` | 环境类型：`atari` 或 `mujoco_playground` | `atari` |
| `training.obs_type` | 观测类型：`image` 或 `state` | `image` |
| `training.seed` | 随机种子 | `1` |
| `training.device` | GPU 设备编号 | `0` |
| `training.total_steps` | 总训练步数 | `102000` |
| `training.batch_size` | 批大小 | `16` |

日志和 checkpoint 保存在 `outputs/` 目录下。

## Evaluation

```bash
python eval.py eval.env_name=CartpoleBalance eval.env_type=mujoco_playground eval.obs_type=state eval.ckpt_path=outputs/.../ckpt/100000 eval.device=0
```

`eval.env_type` 和 `eval.obs_type` 需要与训练时保持一致。

## Experiments

Add comparison with different nets architectures in several atari games. The results will be added in the future.