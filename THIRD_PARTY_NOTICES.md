# 第三方来源与发布范围

本工程原创部分沿用目标仓库的 [MIT 许可证](LICENSE)。这个授权不重新许可外部依赖、行情数据或官方模型。

## 随源码发布的 ICT 文件

`vendor/ict_trading_model/` 来自 [carreypro/ict-trading-model](https://github.com/carreypro/ict-trading-model)，固定 commit 为 `7162437e28b1f59a4b25ea88615ee99902c9cc75`。保留原始 [MIT 许可证](vendor/ict_trading_model/LICENSE)、版权声明与源码；逐文件 SHA 见 [source_manifest.json](vendor/ict_trading_model/source_manifest.json)。本工程在外部适配层实现因果确认规则，没有修改原始文件。

## 通过本地路径引用，不随包发布

- KHQuant：冻结的行情 adapter、量价服务、原生 CZSC 桥和有来源链的交易日历。使用者需另行准备兼容的 KHQuant 工程及其依赖。本仓库未包含 KHQuant 源码。
- Kronos：KHQuant 中冻结的官方 Kronos 源码、`NeoQuasar/Kronos-base` 及 `NeoQuasar/Kronos-Tokenizer-base` 权重。许可证随原始资产保留；本仓库不复制官方源码、大权重或 tokenizer。
- 行情：使用者有权访问的沪深 A 股本地行情。数据库、实际 OHLCVA 窗口和派生逐条标签均未发布。
- NumPy、pandas、PyTorch、safetensors、tqdm、pytest、Matplotlib、CZSC 等依赖分别适用各自的许可证，不包含于源码包。

固定资产布局、版本和校验方式见 [外部依赖说明](docs/external_dependencies.md)。本次发布包含代码、配置、合成测试、方法说明与汇总实验结果；不包含训练检查点、逐样本预测、私有路径清单或运行缓存。
