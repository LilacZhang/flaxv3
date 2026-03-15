import swanlab
from tensorboard.backend.event_processing import event_accumulator

# 1. 配置路径和项目信息
# 这里的 path 应该是包含 'events.out.tfevents...' 文件的文件夹路径
tensorboard_log_dir = "./outputs/2026-03-14/13-53-37" 
project_name = "boxing_experiments_baseline"
experiment_name = "old_tensorboard_data"

# 2. 初始化 SwanLab
swanlab.init(project=project_name, experiment_name=experiment_name)

# 3. 解析 TensorBoard 数据
ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
ea.Reload()

# 获取所有的标量（Scalars）标签
tags = ea.Tags()['scalars']

# 4. 遍历并上传数据
for tag in tags:
    events = ea.Scalars(tag)
    for event in events:
        # event.step 是步数，event.value 是数值
        swanlab.log({tag: event.value}, step=event.step)

print("数据同步完成！")