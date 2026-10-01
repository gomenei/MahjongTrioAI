# 发布范围与本地审阅

这个目录是独立的 GitHub 发布候选，文件来自源代码白名单。它不含原项目 `.git` 目录或提交历史。

## 已保留

- 三人日麻环境、GUI、模型结构、监督训练、PPO 和评测代码。
- Mahjong Copilot 源码、MahjongTrioAI 适配、协议类型及界面必需资源。
- 人工构造的规则/特征测试；依赖真实牌谱 fixture 的测试部分未公开。
- 经过字段白名单过滤的对战 JSON/CSV 汇总，包含未完成标志和桥接后备动作总数。
- 对战结果中每个模型的 `seed_groups` 和每项比较的 `paired_seed_groups` 均至少为 1000；小于 1000 的实验只保留在原项目本地目录。
- README、技术说明、第三方来源、忽略规则和发布清单。

## 已排除

模型权重、数据集与数组缓存、真实牌谱、爬取和远程抓取代码、训练日志、TensorBoard、进度快照、逐局记录、错误消息、服务器状态与连接信息、SSH 配置、API 凭据、实际 Copilot 设置、浏览器用户目录、代理证书和私钥、抓包、虚拟环境、二进制扩展、发行包及上饶麻将模块。

`mahjong_env/deal.py` 和 `gui/replay.py` 是读取用户自备文件的重放代码，未包含牌谱获取逻辑。`liqi_proto/` 是协议类型定义，未包含登录会话或真实数据。

## 检查

```powershell
python scripts/check_public_release.py
python -m pytest -q
```

`PUBLIC_MANIFEST.json` 记录发布文件的路径、字节数和 SHA-256。检查程序会核对清单、文件内容中的高置信凭据模式、私有路径和禁止发布的文件类型；它也能检查压缩包：

```powershell
python scripts/check_public_release.py --archive ../MahjongTrioAI-github-public.zip
```

检查结果只说明列出的检查通过。公开前仍应阅读 README、第三方来源和对战结果中的实验边界。后续修改公开文件后需要同步更新清单；运行时产生的 `.git`、虚拟环境和 Python/pytest 缓存不属于发布清单。

## 审阅后建立新仓库

在此目录运行，而不是在原工作目录运行：

```powershell
git init
git add .
git diff --cached --stat
git diff --cached --name-only
```

确认暂存文件与清单一致后，再提交并配置你自己的 GitHub remote。上传远端由用户审阅后另行进行。

不要直接推送原项目的历史：原暂存区已有模型、数据集和输出文件，`.gitignore` 不会自动排除已跟踪或已暂存的文件。新仓库方式避免把旧历史带入发布。
