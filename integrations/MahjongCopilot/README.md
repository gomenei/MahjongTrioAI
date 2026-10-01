# Mahjong Copilot 集成

本目录包含 [latorc/MahjongCopilot](https://github.com/latorc/MahjongCopilot) 的本地源码，以及本项目的 MahjongTrioAI 三麻策略适配。上游版权声明和 [LICENSE](LICENSE) 原样保留。

本地上游 checkout 的基准提交为 `31be3de972bbe4b1261360ff61b18f1bb84babbe`。当前目录还包含本项目修改，未携带该 checkout 的 Git 元数据。

## 本项目改动

- `bot/mahjongtrio/`：把 MJAI 事件流转换为三麻特征、合法动作与模型输入，并把策略动作转回 MJAI。
- `bot/factory.py`：增加 MahjongTrioAI 模式；公开版按实际选择延迟导入其他机器人，使 MahjongTrioAI 模式无需 Mortal 原生库。
- `common/settings.py`：支持本项目 checkpoint、设备和动作选择设置；公开版默认使用 `model/model.pt`。
- `libriichi3p/__init__.py`：保留按平台/Python 版本寻找扩展的加载源码，未附扩展二进制。
- 根目录 `setup_mahjong_copilot.py` 与 `run_mahjong_copilot.py`：独立环境安装与启动入口。

## 配置

从项目根目录运行安装和启动脚本，具体命令见 [主 README](../../README.md)。将自备模型存到根目录 `model/model.pt`，在设置中选择 `MahjongTrioAI`，或复制 `settings.example.json` 为本目录 `settings.json`。

只提供不含账号和密钥的示例；实际 `settings.json`、浏览器会话、代理证书、日志和在线 API 凭据均被忽略。模型缺失时无法提供本地推理建议，需先放置有效 checkpoint。

MahjongTrioAI 模式支持三人麻将。上游四麻 Mortal、在线模型与进程代理注入仍保留相应源码，但依赖和运行条件需由用户自行配置；发布包不含上游模型、原生库、注入程序、浏览器扩展和截图。默认关闭自动操作、自动入场及进程注入。

`liqi_proto/` 是协议类型定义与生成代码，其中出现的账号、token 等名称是协议字段，不包含真实账号、密钥或抓包记录。
