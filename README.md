# CZSC + ICT → Kronos 知识蒸馏

把 CZSC 结构量价与 ICT 因果原子事实作为离线教师，联合训练同一个 Kronos 学生。官方 tokenizer 和原始主干参数冻结，以外部 q/v LoRA 与内部任务头学习；日常推理目标是输入 OHLCVA 与日期，由学生自身输出估计。

**已接通真实数据生成、蒸馏训练、检查点重载和四组多种子评估。240 股回溯消融已完成：历史教师事实重建改善，稳健的未来方向收益尚未确认。**

- [如何蒸馏：完整操作指南](docs/distillation_guide.md)
- [外部依赖、安装和固定模型身份](docs/external_dependencies.md)
- [240 股消融结果与原因分析](docs/results_240.md)
- [数据与时间契约](docs/data_contracts.md) · [实施记录](docs/implementation_status.md)

## 发布范围

包含研究源码、配置、合成测试、固定版本 ICT 源码及其 MIT 许可证、方法说明和汇总实验结果。行情数据库、逐条行情/标签、官方大权重、tokenizer、训练检查点、缓存与本地路径清单不随包发布。

**真实数据通路依赖兼容 KHQuant。** 本仓库尚无独立 CSV 入口或自动下载器。仅克隆即可运行合成测试和查看计划；实际生成/训练前必须自行准备只读 KHQuant、已验证交易日历和固定官方资产，详见[依赖说明](docs/external_dependencies.md)。

## 安装与检查

建议 Python 3.12，最低 3.11。在仓库根目录执行：

```bash
git clone https://github.com/zhaozhenxiong/token_distillation_kronons.git
cd token_distillation_kronons
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,dev]'
export TD_PYTHON="$PWD/.venv/bin/python"
./scripts/check.sh
./scripts/td.sh train-plan
```

真实资产准备完成后执行 `./scripts/td.sh doctor --verify-assets`。也可将 `TD_PYTHON` 指向既有兼容环境，直接通过 `run.py` 使用，无需修改源工程。

`configs/default.toml` 默认禁用训练。doctor、inspect、plan、合成测试不会启动训练或联网。生成、训练和评估都需要[显式命令](docs/distillation_guide.md)；源码打包和 GitHub CI 不运行真实训练。

## 蒸馏流程

```text
同一因果历史窗口 → W32 / 原生 CZSC / ICT 教师目标与 masks
               → 上下文归一化 → 冻结 tokenizer → Kronos + LoRA
                                                 ├─ 原有 s1/s2 token 头
                                                 ├─ 未来方向头
                                                 └─ W32 / CZSC / ICT 内部头
```

共同目标为自回归 token 损失与真实未来方向 CE；教师连续目标使用 masked Huber，事件使用 masked BCE。未知目标按 mask 排除。四组 `none / czsc / ict / dual` 共用样本、初始化、抽样顺序、内部头和预算，仅改变教师损失。正式消融关闭基座 KL，三个教师组辅助系数总和均为 0.6。

原生 CZSC 是在学生上下文内重新初始化的已确认结构；ICT 只学习经确认时点检查的日线原子事实。完整买卖点、多周期、纽约时段、OB/Breaker/OTE 等不作为已完成监督。学生输出属于估计，不能标记成原生已确认事实。

## 240 股结果

2023 截面分层冻结 240 股；实际训练 239 股、留出 237 股。128 根上下文、未来 5 根、步长 5；29,729 个窗口，train 16,611、holdout 5,326，留出只有 24 个预测日期。四组 × 三种子，每次 8,306 步，均由 tune 选中第 4,154 步。

| 配置 | 留出 LogLoss ↓ | Brier ↓ | 准确率 |
| --- | ---: | ---: | ---: |
| 无教师 | 0.928067 | 0.584206 | 45.29% |
| CZSC | 0.927989 | 0.584176 | 45.31% |
| ICT | 0.927691 | 0.583942 | 45.44% |
| 双教师 | 0.927948 | 0.584141 | 45.32% |
| 训练频率先验 | 0.908154 | 0.565831 | 49.49% |

教师组主要日期块差值区间均跨零，所有组仍弱于训练先验；80 股真实双教师与打乱教师的方向评分近乎相同。当前支持内部历史知识重建，未证明双教师语义带来未来预测优势。见[完整结果及限制](docs/results_240.md)与[机器可读汇总](docs/results/ablation240_summary.json)。

本机 M3 Max / 36 GiB / MPS 的正式 12 次训练与评估实测 4 小时 29 分钟；包括缓存和 80 股前置对照的 study 为 5 小时 27 分钟，数据生成另计。1,200 股与全量尚未启动。

## 目录

```text
configs/                 默认、烟测、20 股与 240 股研究配置
src/token_distillation/  只读适配、数据契约、教师、LoRA、训练及评估
scripts/                 分层/活跃选样、分片生成、研究编排、诊断和源码打包
tests/                   临时目录与合成数据测试
schemas/                 来源、数据集、教师及学生清单 schema
vendor/ict_trading_model/ 固定 commit/SHA 的 ICT 原文件及 MIT 许可证
docs/                    操作指南、依赖、研究结果、历史设计与实施记录
audit/                   可公开的变更事实；本地运行清单不提交
```

源码也包含尚未纳入本次结果验收的 800 股活跃样本扩展入口，见[说明](docs/distillation_guide.md#9-活跃样本扩展入口未作为本次发布完成结论)。历史设计文档描述当时的计划，当前能力以操作指南和实施记录为准。尚未实现优化器恢复、概率校准、交易政策拟合、walk-forward 资格验收、独立合并模型导出及自动晋升。条件 token NLL 不是自由生成价格路径或交易收益。

## 打包与许可证

```bash
python scripts/package_source.py --output artifacts/publication/token_distillation_kronons-0.1.0.zip
```

打包只读取允许清单，附逐文件 SHA-256，不生成行情或训练模型。项目沿用目标仓库的 [MIT 许可证](LICENSE)，第三方来源与发布边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
