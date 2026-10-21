# 外部依赖与运行环境

本仓库是独立的蒸馏研究工程，但真实行情通路仍依赖兼容的 KHQuant。仅克隆本仓库可以运行合成测试、查看训练计划与阅读结果；不能在缺少外部资产时直接训练。没有通用 CSV 数据入口，也没有自动下载或替代数据。

## 环境

建议使用 Python 3.12。项目声明最低 Python 3.11；正式 240 股实验在 macOS、M3 Max、36 GiB 内存、MPS 上完成。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,dev]'
export TD_PYTHON="$PWD/.venv/bin/python"
./scripts/check.sh
```

这里只安装本工程声明的依赖。真实教师还需要兼容 KHQuant 的依赖，尤其原生 CZSC。也可将 `TD_PYTHON` 指向已经验证过的 KHQuant Python 环境，直接使用 `run.py`，无需向该环境安装本包。

240 股运行环境观测版本：Python 3.12，NumPy 2.5.3，pandas 3.0.6，PyTorch 2.13.0，safetensors 0.6.2，tqdm 4.70.1，CZSC 1.0.1，pytest 9.1.1。这是当时的环境记录，不是跨硬件位级复现保证。CUDA 用户需自行安装适配其平台的 PyTorch；显式指定不可用的 MPS/CUDA 会报错。

## KHQuant 布局

默认假设本工程与 KHQuant 相邻：

```text
workspace/
├── token_distillation_kronons/
└── KHQuant/
    ├── .venv/bin/python
    └── app/
        ├── my_strategy/
        │   ├── adapters/czsc_adapter.py
        │   ├── services/czsc_price_volume_features.py
        │   ├── services/czsc_price_volume_events.py
        │   ├── services/czsc_research_calendar.py
        │   ├── services/czsc_trend_support.py
        │   ├── services/kronos_assets.py
        │   ├── core/paths.py, run_context.py, tz.py
        │   ├── runtime_env.py
        │   ├── data/raw/khquant_raw.db
        │   ├── artifacts/runs/<run_id>/reports/calendar.json
        │   └── kronos_model/
        │       ├── config.json, model.safetensors
        │       └── tokenizer/config.json, model.safetensors
        └── vendor/kronos/
            └── module.py, kronos.py, LICENSE, source_manifest.json
```

交易日历必须具有可验证的来源和父链。任意日期列表不能冒充冻结日历。SQLite 读取表 `stock_daily_normalized`；基础字段为 `stock/date/open/high/low/close/volume/amount`，真实教师与人口筛选还会验证来源、价格有效性和可见时间等字段。只准备一个同名 SQLite 表不能替代完整 adapter 契约。

可用环境变量指定本机路径；脚本不读取 `.env`：

```bash
export TD_SOURCE_APP="/path/to/KHQuant/app"
export TD_RAW_DB="/path/to/KHQuant/app/my_strategy/data/raw/khquant_raw.db"
export TD_KRONOS_ROOT="/path/to/KHQuant/app/my_strategy/kronos_model"
export TD_SOURCE_PYTHON="$TD_PYTHON"
export TD_CALENDAR_PATH="/path/to/KHQuant/app/my_strategy/artifacts/runs/RUN_ID/reports/calendar.json"
```

`TD_PYTHON` 决定 shell 脚本使用的解释器；`TD_SOURCE_PYTHON` 是配置中的外部解释器引用，两者含义不同。配置中的相对路径均相对于仓库根目录解析。输出只允许在本工程 `data/`、`artifacts/`、`logs/`、`audit/` 下；不能把源工程设为输出目录。

## 官方资产的固定身份

运行时严格检查字节，不使用任意同名模型替换：

| 资产 | 固定 revision / SHA-256 |
| --- | --- |
| Kronos-base revision | `2b554741eca47781b64468546e77fef3e85130e6` |
| Kronos-base model.safetensors | `abff193acab6db1a0368e9773e75799d11403b6d054ee6d5f0a11aeabc5f4b83` |
| Tokenizer-base revision | `0e0117387f39004a9016484a186a908917e22426` |
| tokenizer/model.safetensors | `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee` |

完整 config/vendor 清单与加载逻辑见 [`models/runtime.py`](../src/token_distillation/models/runtime.py)。更换模型版本或 KHQuant 实现需要另建明确版本的适配与验证流程，不能通过删除哈希检查复用旧结果。

```bash
./scripts/td.sh doctor --verify-assets
./scripts/td.sh sources --verify-assets --output data/references/sources.local.json
```

第二条写入本地来源清单；它不应提交到公开仓库。数据库通过 `mode=ro` 和 `query_only` 读取；完整数据库不做隐式复制，实际载入窗口另存内容身份。源库文件大小/mtime 相同不等于完整内容哈希一致。

## 源码发布包

```bash
python scripts/package_source.py --output artifacts/publication/token_distillation_kronons-0.1.0.zip
```

该命令按允许清单打包代码、文档、合成测试、配置、ICT 原文件和公开审计。只读取派生目录中的 `.gitkeep`，拒绝符号链接，并附逐文件 SHA-256 清单。它不生成数据、不导入教师、不加载模型或启动训练。
