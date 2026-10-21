# token_distillation 工程约定

- 这是独立的 CZSC 量价知识到 Kronos 蒸馏工程。代码与派生输出只写在本工程内。
- KHQuant 源码、行情数据库、日历、官方模型与 tokenizer 均通过显式路径只读引用；不能复制大权重、写源库、覆盖现任模型或恢复旧任务。
- 数据访问优先经过冻结 KHQuant adapter；SQLite 必须 mode=ro 且 query_only。外部模块仅在需要时延迟加载，隔离其输出路径。
- 只使用沪深 A 股 sh_sz_a_v1。严格区分 as_of、available_at、dependency_start、label_end、label_mature_at；未知不能当无事件。
- 归一化仅使用预测原点之前的上下文；训练/调参/校准/政策/留出按时间切分，真实标签跨界须清洗。
- 官方 tokenizer 冻结；学生主干以外部 LoRA 注入，保留原始 vendor。学生内部输出与原生 CZSC 事实分别标记。
- ICT 来源固定为 vendor/ict_trading_model/source_manifest.json 的 commit 与文件 SHA；原始代码保持只读并保留 MIT 许可证。仅经确认时点核验的原子特征进入监督；原 scanner 的机械 confirmed、纽约时段及未具备的多周期证据不能直接当沪深日线交易标签。
- 新训练和批量生成必须通过显式命令。doctor、inspect、plan 和测试不得启动训练、联网、晋升模型或产生源工程业务状态。
- 本轮工程搭建范围内，训练/评估/导出若尚未实现，必须明确报未实现，不能生成成功报告或假模型。
- 测试使用临时目录及合成数据；真实只读烟测另行记录。显式请求的 MPS/CUDA 不可用时应报错。
- 每次变更在 audit/ 追加事实记录；本轮 KHQuant 关联审计会话为 20261019-134115-83f5b455。
