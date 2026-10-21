# 如何把 CZSC 与 ICT 知识蒸馏进 Kronos

## 1. 学什么、更新什么

这是规则知识的多任务蒸馏：教师在离线数据生成阶段读取同一个历史窗口，输出结构量价目标。学生用这些目标训练内部头，并通过共同损失更新 Kronos 的外部 LoRA。CZSC/ICT 不是输出 token 概率的神经教师。

```text
已核验的同一历史 OHLCVA 窗口
 ├─ W32 量价教师 ──────────── 25 连续目标 + 7 事件 + masks
 ├─ 原生 CZSC 窗口教师 ────── 16 个已确认笔/中枢目标 + masks
 ├─ ICT 因果原子教师 ──────── 2 连续目标 + 6 事件 + masks
 └─ 仅用上下文计算归一化
       └─ 冻结官方 tokenizer → Kronos + q/v LoRA
                                  ├─ 方向头：未来 5 根收益三分类
                                  ├─ W32 / CZSC / ICT 内部头
                                  └─ 原有 token 预测：s1 / s2
```

官方 tokenizer 与主干原始参数冻结，只更新 transformer 自注意力 `q_proj/v_proj` 的 LoRA 及内部头。240 股配置为 rank 4、alpha 8；内部头共同使用末位因果隐藏状态及 20 项只由上下文计算的归一化侧信息。统计量不会输入未来数据。

W32 原有 22 个结构槽仍 masked，原生 CZSC 结构由独立 schema 提供。ICT 只监督可核验确认时点的 swing 距离、FVG、sweep 和收盘穿越；完整 setup、纽约时段、OB/Breaker/OTE、多周期等不作为沪深日线已确认标签。学生内部输出必须标记为估计。

## 2. 联合损失与消融

共同目标为 `L_AR_s1 + L_AR_s2 + L_direction`。方向标签使用原点收盘至第 5 根未来收盘收益，按 ±0.5% 分为下跌、平盘、上涨。未来 token 的 AR 目标使用因果 teacher forcing，s2 依赖层通过外部适配保持因果性。

连续教师目标使用只在 train 拟合的缩放统计与 masked Huber；事件用 masked BCE。每项按有效元素数归一化，未知 mask 不能填成负例。

| 模式 | W32 系数预算 | 原生 CZSC 系数预算 | ICT 系数预算 |
| --- | ---: | ---: | ---: |
| `none` | 0 | 0 | 0 |
| `czsc` | 0.30 | 0.30 | 0 |
| `ict` | 0 | 0 | 0.60 |
| `dual` | 0.15 | 0.15 | 0.30 |

分支内部再按配置分配连续/事件权重；默认有两类目标的分支各半，原生 CZSC 只有连续项。所有组保留相同内部头和共同目标。辅助系数总和相等不意味着梯度大小或信息量相等。代码支持冻结基座 KL 保持项，但正式四组消融统一显式设为 0，`none` 没有辅助教师损失。

## 3. 先准备外部资产和检查环境

详见 [外部依赖](external_dependencies.md)。以下命令在仓库根目录、兼容环境中执行：

```bash
git clone https://github.com/zhaozhenxiong/token_distillation_kronons.git
cd token_distillation_kronons
# 安装本工程依赖，并按外部依赖文档设置 TD_PYTHON / TD_SOURCE_APP 等。
./scripts/check.sh
./scripts/td.sh doctor --verify-assets
./scripts/td.sh train-plan
```

合成测试和 `train-plan` 不需要真实行情或权重，不启动训练。doctor 不训练、不联网，但需要真实来源完成检查。默认 `configs/default.toml` 禁用训练；训练必须显式选择 `execution_enabled=true` 配置及执行命令。

## 4. 先跑三股通路验证

`configs/smoke.toml` 固定 128 根上下文、5 根未来、MPS 与五个时间阶段。输出目录选新的名字，以下示例依次生成、训练、重载评分：

```bash
./scripts/td.sh --config configs/smoke.toml build-dataset \
  --symbols 600519.SH 600036.SH 000001.SZ \
  --start 2026-01-05 --end 2026-10-09 --label-through 2026-10-16 \
  --stride 5 --max-per-split 24 --output-dir data/smoke_demo

./scripts/td.sh --config configs/smoke.toml train \
  --dataset data/smoke_demo --output-dir artifacts/runs/smoke_demo \
  --teacher-mode dual --device mps --max-steps 24 \
  --gradient-accumulation-steps 1

./scripts/td.sh --config configs/smoke.toml evaluate \
  --dataset data/smoke_demo \
  --checkpoint artifacts/runs/smoke_demo/adapter_checkpoint.pt \
  --output-dir artifacts/reports/smoke_demo --split all --device mps
```

这里的未来行情必须真实存在且标签成熟。例子使用研究期日期，不能自动滚动到“今天”。支持 `--device cpu/cuda` 的通用命令可按设备显式替换；不能请求 MPS 后静默回退。

## 5. 240 股分层数据与完整研究

`configs/ablation240.toml` 使用以下回溯切分：

| 阶段 | 日期范围 | 用途 |
| --- | --- | --- |
| train | 2024-01-01 至 2025-06-30 | 更新权重、拟合教师缩放/常数 |
| tune | 2025-07-01 至 2025-10-31 | 按方向 LogLoss 选检查点 |
| calibration | 2025-11-01 至 2025-12-31 | 隔离报告，未拟合校准器 |
| policy | 2026-01-01 至 2026-03-31 | 隔离报告，未拟合交易政策 |
| holdout | 2026-04-01 至 2026-10-09 | 留出报告，不参与选点 |

严格区分 `as_of / available_at / dependency_start / label_end / label_mature_at`。标签跨阶段或尚未成熟的窗口被清洗。128 根上下文包含 W32 所需的 120 根预热；不能随意缩短后把未知项当作无事件。

下面是已有研究的重跑入口，当前脚本固定 240/80 股、研究日期、默认产物路径及 MPS。它们不是 1,200 股或全市场通用入口。以下命令均实际执行对应阶段，应逐条运行：

```bash
# 仅用 2023 截面筛选：板块 × 成交额 × 波动率，36 层。
"$TD_PYTHON" scripts/profile_local_universe.py \
  --config configs/ablation240.toml \
  --output-dir artifacts/reports/population240_20261021

# 每片 8 股，最多 4 个 CPU 进程；生成主池与嵌套 80 股数据。
"$TD_PYTHON" scripts/build_stratified_dataset.py \
  --config configs/ablation240.toml \
  --population artifacts/reports/population240_20261021/population.json \
  --output data/ablation240_20261021 --workers 4

# 编码缓存 → 80 股机制控制 → 240 股四组 × 三种子 → 重载评估。
"$TD_PYTHON" scripts/run_240_study.py \
  --config configs/ablation240.toml --dataset data/ablation240_20261021 \
  --output-dir artifacts/runs/study240_20261021
```

人口选择只用训练前数据，不按之后覆盖或表现替换股票。24 股压力组只用于边界检查，不混入效果评价。派生数据过滤保留父分片和排除清单；其中未来标签区间的疑似未复权跳空筛查属于事后质量准入，不能当成预测原点可用的筛股条件。

脚本拒绝覆盖既有目录。仅数据分片可以通过显式 `--continue-chunks` 并验证身份后复用；这不等于恢复优化器。完整重跑建议使用新的仓库工作副本；修改产物路径时还须同步配置中 `token_cache` 和脚本相关报告路径。不要把新的来源或配置套在旧缓存/检查点上。

主流程自动按 `ceil(2 × train_windows / 4)` 计算更新预算，评估间隔为 `ceil(max_steps / 4)`，包括 step 0 共约 5 次 tune。240 股实测 16,611 个 train 窗口，对应每组 8,306 步。所有组完整执行预算后，由 tune 独立选点。

## 6. 单独执行通用四组消融

以下适用于已经生成且绑定同一配置的数据集。若配置开启 token_cache，应先显式生成匹配缓存。240 股已完成的 study 不要再运行到相同目录：

```bash
"$TD_PYTHON" scripts/prepare_token_cache.py \
  --config configs/ablation240.toml --dataset data/ablation240_20261021 \
  --output-dir data/ablation240_20261021_token_cache --device mps

./scripts/td.sh --config configs/ablation240.toml ablation-plan \
  --seeds 42 123 2026 --max-steps 8306 --batch-size 4 \
  --evaluation-interval 2077 --evaluation-batch-size 32

./scripts/td.sh --config configs/ablation240.toml run-ablation \
  --dataset data/ablation240_20261021 \
  --output-dir artifacts/runs/ablation240_manual \
  --seeds 42 123 2026 --max-steps 8306 --batch-size 4 \
  --evaluation-interval 2077 --evaluation-batch-size 32 --device mps

./scripts/td.sh --config configs/ablation240.toml analyze-ablation \
  --dataset data/ablation240_20261021 \
  --experiment-dir artifacts/runs/ablation240_manual \
  --output-dir artifacts/reports/ablation240_manual
```

同种子组核验初始化和实际抽样顺序 SHA；样本、LoRA 容量、总更新预算和选点指标共同固定。首次缓存与完整 study 内的缓存阶段二选一，不能对相同路径重复生成。

## 7. 读懂产物、判断是否有效

- 数据集：`manifest.json` 与五段 JSONL，保留样本、确认时点、masks、来源/日历/schema/内容哈希。
- 单次训练：`training_summary.json`、选中的 `adapter_checkpoint.pt`，包含 LoRA、内部头、优化器及身份；没有官方大权重副本。
- 消融运行：`experiment_manifest.json`、各组各种子的训练与独立重载评估、配对初始化及抽样检查。
- 完整研究：`study_status.json` 记录各阶段状态。失败不能解释为完成；只有全部 12 组完成才生成主比较。
- 主分析：方向 LogLoss/Brier、训练频率先验、同种子无教师差值、日期块区间、教师重建相对训练常数、条件 token NLL。

“有效”要分开回答三件事：历史教师事实是否重建得更好；教师语义是否经由主干影响结果；未来方向是否改善且超过简单基线。增加样本行数不增加独立市场日期。按预测日期连续块重采样时，同日股票和固定种子放在一起，并报告块长敏感性。

80 股脚本提供 `--mechanism-control shuffle` 和 `frozen_backbone`，分别检查打乱 train 教师值及只训练头。补充留出评分脚本 `evaluate_mechanism_controls.py` 还要求事前冻结的协议与检查点 SHA；本地协议不随源码包发布，克隆后不能把缺少协议说成已经复现。

[240 股结果](results_240.md) 表明知识重建已有进展，但未来方向收益尚未稳健确认。AR NLL 是条件 teacher forcing 评分，不能解释为自由生成价格精度或交易收益。

## 8. 边界和扩展

恢复优化器训练、概率校准、交易政策拟合、滚动资格验证、独立合并导出及自动晋升尚未实现。检查点仍依赖原始官方资产和身份一致的运行环境，不是可独立部署的全量模型包。

扩到 1,200 股时，需要冻结新的训练前量价筛选、板块分层、人口及专用配置，并将步数按有效 train 窗口重算。当前仅有耗时估算，没有生成或训练 1,200 股。保持两遍、四组 × 三种子的本机排期估算为 26–32 小时，未做该规模内存验收。


## 9. 活跃样本扩展入口（未作为本次发布完成结论）

源码快照另外包含 `configs/active800.toml`、`scripts/select_active_population.py` 和 `scripts/run_active_study.py`。该扩展选择器只用 2023 截面的成交额、收益波动、振幅和量能变化，当前配置排除科创板与北交所；默认 800 股。它与 240 股的 36 层样本政策不同，不能混用结果或直接沿用逆入样权重。

扩展编排要求显式提供新人口、配置、数据/运行/报告目录及事前 `--freeze` 清单，逐阶段核验 `file_hashes`。本次发布只验收其合成选择测试，不发布该扩展的真实训练结果、来源冻结清单或完成声明。240 股结果不构成该扩展已完成的证据；需要另行形成可复现协议和完成报告。
