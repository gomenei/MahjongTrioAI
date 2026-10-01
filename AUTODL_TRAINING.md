# 计算服务器训练

这里仅保留训练编排源代码。原项目的服务器连接信息、任务状态、SSH 工具、上传包和日志没有公开。

将源代码与自己准备的数据、checkpoint 分别传入服务器。在已安装匹配 GPU 的 PyTorch 环境中安装其他依赖，再运行 `ppo_train.py` 或候选规则入口。CUDA 的 PyTorch 版本应由运行者按自己的设备环境确定。

`run_autodl_training.py` 提供 `smoke`、`train`、`screen`、`confirm` 和 `bundle` 阶段；其他 `run_autodl_*.py` 提供历史实验编排。它们的默认基础模型与历史对手路径源于本地研究，运行前需检查 `BASE`、`HISTORY` 等常量和命令行参数。

`bundle` 会把本地权重一起打入服务器训练包，这是私人训练工作流，不能作为 GitHub 发布包。GitHub 发布使用这里的纯源代码目录，并运行 `scripts/check_public_release.py`。
