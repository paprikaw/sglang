# SGLang 优化版本追踪简报

日期：2026-10-31。

**此前已有本地 Git 提交和实验标签，但远端备份不完整，部分改动仍未提交。现在已补齐源码提交并推送 GitHub。**

## 已完成

| 仓库 | 本次核对并推送的范围 |
| --- | --- |
| [paprikaw/sglang](https://github.com/paprikaw/sglang) | 83 个分支、42 个原有实验标签；另增加 1 个本次归档标签 |
| [paprikaw/sglang-cpu-kv-decode-replay](https://github.com/paprikaw/sglang-cpu-kv-decode-replay) | 17 个实验工具分支 |
| [paprikaw/experiment-guard](https://github.com/paprikaw/experiment-guard) | 1 个缓存适配代码快照分支 |

保留原有提交历史，补充 **15 个源码提交和 1 个归档说明提交**。覆盖当前未提交的引擎改动、PP/EP 优化、HiCache/GDN、PD 状态传输、AgentX replay/公平分流，以及 10 月 20–31 日实验源码和各轮 frozen code。源码选择清单为 10,187 个文件，重复 frozen code 由 Git 对象存储去重。

DeepEP 的部分历史远端原先只指向本机目录；现已备份到 SGLang 仓库的 `codex/dependency-deepep/*` 分支。这些分支保存 **DeepEP 自身源码**。122B 的三个独立优化代码仓库也已保留为对应的 `codex/qwen122b-*` 分支。

## 主要版本入口

| 用途 | 版本／分支 |
| --- | --- |
| 当前 AgentX 引擎 R8，混合 Attention 的 PP KV 层映射 | [`fa52c3cd95d2`](https://github.com/paprikaw/sglang/commit/fa52c3cd95d2c171c9eac9a8bfdc32296aa8c75f) |
| 历史 AgentX 引擎 R5，PP HiCache 层映射 | [`37e0828c642c`](https://github.com/paprikaw/sglang/commit/37e0828c642ce4f3795c4c4394f4ce913bcd1a35) |
| EP Decode overlap | [`0f2f19909947`](https://github.com/paprikaw/sglang/commit/0f2f1990994774472b1089d2b0682f0bf926a6b4) |
| 实验源码、适配器、frozen code 和报告汇总 | [`codex/sglang-optimization-records-20261031`](https://github.com/paprikaw/sglang/tree/codex/sglang-optimization-records-20261031) |
| 待进一步验证的开发改动：PD PP1→PP2、SM80 Verify | [`codex/archive/sglang-development-20261031`](https://github.com/paprikaw/sglang/tree/codex/archive/sglang-development-20261031) |
| Elastic PP 的 KV fence 修复快照 | [`codex/archive/elastic-pp-fence-20261031`](https://github.com/paprikaw/sglang/tree/codex/archive/elastic-pp-fence-20261031) |
| 尚未完成资格验证的 TBO 尝试 | [`codex/archive/qwen35-ep2-tbo-wip-20261031`](https://github.com/paprikaw/sglang/tree/codex/archive/qwen35-ep2-tbo-wip-20261031) |

R1–R8 保持独立版本；补齐了 R6 detached HEAD 和 R7 非标准分支名对应的 `codex/` 入口。引擎 R/E 版本与 replay V3/V4 分别追踪，不能仅凭其中一个判断实验实现。

## 核验与边界

- 已逐项比较推送分支／标签的本地 SHA 与远端 SHA；三个仓库的 `main` 均保持原 SHA。新增归档标签为 `archive/sglang-optimizations-20261031`。
- 原实验工作树、源代码内容和已有实验标签保持原状。其未提交改动已逐字节保存到独立快照分支，所以原工作树仍可能显示 dirty；归档分支包含对应内容和文件 SHA256。
- 28 个待归档 Python 文件通过语法检查；公平路由 6 项、Decode 状态预算 4 项 CPU 检查通过。Branch timing 测试在当前两个 Python 环境中分别受 AgentX 插件枚举和 `ruamel` 缺失阻碍，未计为通过。本次没有新增 GPU 测试，WIP 归档不代表性能或正确性资格验证通过。
- 模型权重、KV/GDN 缓存、大型日志和 profiler payload 仍保留在 GPFS，Git 保存源码与已有报告。第一次推送发现 DeepEP 浅克隆缺少祖先对象，补齐原历史后重试成功，未改写源提交。

完整可核查记录：[`commit_index.json`](commit_index.json)、[`new_commits.json`](new_commits.json)、[`remote_verification.json`](remote_verification.json)、[`source_selection.json`](source_selection.json)、[`cpu_validation.json`](cpu_validation.json)。`remote_verification.json` 记录源码批次；本报告提交与归档标签另见最终推送回执 `final_receipt.json`。

## 后续约定

1. 每项功能在 `codex/<主题>` 分支开发，一次完整改动连同必要的回归检查形成一个 commit，完成后及时 push。
2. 每次正式实验记录引擎 commit、replay/适配器版本、不可变源码标签和启动参数；资格验证状态单独记录，已有标签不移动。
3. 引擎之外的临时适配器、benchmark 和分析代码也必须提交；运行生成的大文件按路径和哈希关联。汇总分支的忽略规则用于避免误提交运行数据。

本报告已加入 [实验总索引](../README.md)。
