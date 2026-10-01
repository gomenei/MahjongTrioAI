# 第三方来源

| 组件 | 来源 / 状态 |
|---|---|
| `integrations/MahjongCopilot/` | 来自 [latorc/MahjongCopilot](https://github.com/latorc/MahjongCopilot) 的本地源码副本，含项目适配和公开版调整。保留上游版权声明与随附 [LICENSE](../integrations/MahjongCopilot/LICENSE)。 |
| Mortal 推理网络与接口代码 | 随 Copilot 副本保留；相关项目包括 [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) 与 [shinkuan/Akagi](https://github.com/shinkuan/Akagi)。本包不含模型和原生扩展。 |
| `liqi_proto/` | 随 Copilot 保留的协议定义与生成源码；`liqi.py` 保留其原有来源注释。 |
| `assets/tiles/` | 来自当前项目既有牌面素材，只保留 GUI 使用的单张 PNG。原目录未提供单独的素材许可证，本次未新增授权声明。 |
| PyTorch、NumPy、Pygame、mahjong、Playwright 等 | 由依赖文件声明，不将这些库的安装目录或二进制打入仓库。 |

项目原创代码目前沿用原项目的许可状态，本次整理未替其选择新许可证。上游及第三方文件的许可证与版权声明不因目录整理而改变。
