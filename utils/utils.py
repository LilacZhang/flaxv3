from typing import Dict

import numpy as np
import swanlab

class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def log_dict(self, metrics: Dict):
        logged = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) and value.ndim >= 3:
                logged[key] = swanlab.Image(value)
            else:
                logged[key] = float(value)
        swanlab.log(logged)

if __name__ == '__main__':
    # import numpy as np

    # logger = Logger('./logdir')
    # env = gym.make('ALE/Pong-v5')
    # obs,_ = env.reset(seed=5)
    # video = []
    # for _ in range(100):
    #     video.append(obs)
    #     obs,_,_,_,_ = env.step(env.action_space.sample())

    # obs = rearrange(obs,'H W C -> C H W')/255.
    # imageio.mimsave('res.gif',video,fps=20)
    # metrics = {'worldmodel/rep_loss':0.01,'worldmodel/dyn_loss':0.02,'play/image':obs}
    # logger.log_dict(metrics)
    pass
