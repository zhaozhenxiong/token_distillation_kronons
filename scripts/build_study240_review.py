#!/usr/bin/env python3
"""Evidence-linked narrative for the completed study; no model or source IO."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/reports/ablation240_20261021'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    inputs = {
        'main': OUT / 'ablation_comparison.json',
        'strata': OUT / 'stratified_comparison.json',
        'mechanism': ROOT / 'artifacts/reports/mechanism80_holdout_20261021/comparison.json',
        'independent': ROOT / 'artifacts/reports/ablation240_independent_20261021.json',
        'variation': ROOT / 'artifacts/reports/ablation240_prediction_variation_20261021.json',
        'population': ROOT / 'audit/20261021_ablation240_population_postcheck.local.json',
        'source': ROOT / 'audit/20261021_ablation240_source_postcheck.local.json',
        'study': ROOT / 'artifacts/runs/study240_20261021/study_status.json',
        'experiment': ROOT / 'artifacts/runs/study240_20261021/ablation240/experiment_manifest.json',
    }
    d = {k: read(v) for k, v in inputs.items()}
    assert d['study']['status'] == d['experiment']['status'] == d['mechanism']['status'] == 'completed'
    assert d['source']['all_referenced_file_hashes_match']
    h = d['main']['stages']['holdout']
    for a in ('none', 'czsc', 'ict', 'dual'):
        for metric in ('log_loss', 'brier', 'accuracy'):
            assert abs(h['arms'][a][metric]['mean'] - d['independent']['arms'][a][metric]['mean']) < 1e-12
        assert abs(d['strata']['holdout']['arms'][a]['inverse_probability_weighted_logloss'] - d['independent']['arms'][a]['weighted_log_loss']) < 1e-12
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>240股消融研究结论</title><style>body{font:16px/1.7 system-ui;max-width:1160px;margin:32px auto;padding:0 24px;color:#172536}h1{line-height:1.3}h2{margin-top:36px}.lead{font-size:20px;background:#edf4f7;padding:20px;border-left:5px solid #267982}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}td,th{padding:9px;border:1px solid #d5dee5;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}img{max-width:100%}a{color:#186e96}.note{color:#526270}</style>',
             '<h1>240股消融：事实重建有效，稳健预测收益尚未确认</h1>',
             '<p class="lead">CZSC、ICT 连续事实和事件的内部重建已超过训练常数基线。ICT 单教师的方向改善最大；三种教师的主要配对区间都跨过零，所有配置仍弱于训练频率先验。双教师尚未显示优于单教师或打乱控制的预测优势。</p>',
             '<p><a href="index.html">完整四组比较</a> · <a href="stratified.html">36层与板块分析</a> · <a href="../mechanism80_holdout_20261021/index.html">80股机制留出对照</a> · <a href="review_metadata.json">本页输入与哈希</a></p>',
             '<h2>实际样本与预算</h2><p>从2023年本地观测到的4,996股中，4,676股满足主池条件；按板块、成交额和波动率分36层冻结240股，沪主板81、深主板72、创业板62、科创板25。样本选择只使用2023年及之前数据，不按未来覆盖或收益替补。另选24股做边界压力检查，不加入效果汇总。</p>',
             '<p>共有29,729个准入窗口，五段行数为16,611 / 3,649 / 1,633 / 2,510 / 5,326。实际各段覆盖239 / 231 / 238 / 230 / 237股；603958.SH在训练段没有准入窗口，留出段缺少002718.SZ、300068.SZ、603429.SH。上述股票未被事后替补。留出原点2026-04-08至2026-09-22，标签最晚成熟于2026-09-30。</p>',
             '<p>主实验4配置×3种子，每组8,306步、batch 4，约两遍训练数据，总计99,672次更新；12组均由共同tune规则选中第4,154步。80股嵌套机制对照7组各2,766步，共19,362次更新。上下文128根、未来5根日线；官方tokenizer冻结，外部q/v LoRA，各组统一提供20项因果上下文统计。所有29,729窗的token前缀一致性已核验。</p>',
             '<h2>未来方向：留出均值</h2><p>LogLoss与Brier越低越好。差值相对同种子无教师；区间保留同日所有股票及固定种子，使用2个预测原点的连续日期块。</p>',
             '<div class="scroll"><table><tr><th>配置</th><th>LogLoss</th><th>Brier</th><th>准确率</th><th>Δ LogLoss</th><th>主要95%描述区间</th></tr>']
    names = {'none': '无教师', 'czsc': 'CZSC', 'ict': 'ICT', 'dual': '双教师'}
    for a in names:
        m = h['arms'][a]
        e = h['comparisons'][a]['vs_none']['log_loss']
        ci = e['date_block_uncertainty']['ci95']
        parts.append(f'<tr><td>{names[a]}</td><td>{m["log_loss"]["mean"]:.6f}</td><td>{m["brier"]["mean"]:.6f}</td><td>{m["accuracy"]["mean"]:.2%}</td><td>{e["mean"]:+.6f}</td><td>[{ci[0]:+.6f}, {ci[1]:+.6f}]</td></tr>')
    prior_accuracy = h['actual_class_counts'][max(range(3), key=lambda i: h['train_prior'][i])] / h['rows']
    parts.append(f'<tr><td>训练频率先验</td><td>{h["prior_log_loss"]:.6f}</td><td>{h["prior_brier"]:.6f}</td><td>{prior_accuracy:.2%}</td><td>—</td><td>训练段Laplace频率</td></tr></table></div>')
    parts.extend(['<p>ICT三个种子均略有改善，主要LogLoss区间仍跨0；改用5原点块时区间约[-0.000678, -0.000039]，但Brier区间仍跨0。块长敏感性与仅24个日期限制了结论，不能挑选其中一个区间宣称稳健胜出。逆入样概率加权下ICT差值约-0.000417，主要区间仍跨0；权重只对应本地2023主池中具有准入窗口的条件人口。</p>',
                  '<img src="figures/ablation_transfer_matplotlib.png" alt="教师相对无教师的配对差值与日期块区间">',
                  '<h2>知识重建：本轮已有明确进展</h2><p>下表为双教师所选检查点的三种子均值。连续目标比较训练中位数，事件BCE比较训练频率常数；负数更好。未知目标只按mask排除。教师重建没有保存逐行概率，表中不提供日期块区间。</p>',
                  '<table><tr><th>教师事实</th><th>连续归一化MAE Δ 中位数</th><th>事件BCE Δ 频率</th></tr>'])
    for t in ('czsc', 'w32', 'ict'):
        r = d['main']['teacher_acquisition_holdout']['dual'][t]
        event = r['event_bce_minus_constant']['mean']
        parts.append(f'<tr><td>{t}</td><td>{r["continuous_mae_normalized_minus_train_median"]["mean"]:+.6f}</td><td>{"—" if event is None else format(event,"+.6f")}</td></tr>')
    parts.extend(['</table><p>与旧20股试验不同，本轮W32和ICT事件也已超过频率常数，不能继续用“事件完全未学会”解释全部差距。本轮同时改变样本、输入统计和更新数，跨轮变化不能单独归因于扩大样本。</p>',
                  '<h2>机制控制：重建与未来转移不同</h2><table><tr><th>80股配置</th><th>选中步数</th><th>留出LogLoss</th></tr>'])
    for a, r in d['mechanism']['metrics'].items():
        parts.append(f'<tr><td>{html.escape(a)}</td><td>{r["selected_optimizer_step"]}</td><td>{r["direction"]["log_loss"]:.9f}</td></tr>')
    parts.extend(['</table><p>真实双教师减打乱教师标签的LogLoss仅约-0.000000319，2原点块区间约[-0.000144, +0.000151]。两者的预测几乎相同，而真实教师的重建明显更好，当前不能把双教师未来方向收益归因于教师语义。</p>',
                  '<p>冻结主干控制也能在连续事实重建上超过中位数，但事件尚未胜过频率常数。其方向LogLoss点估计较好，区间很宽且选中步数与可训练主干不同，不能据此断言冻结主干普遍更优。双教师重复一次的tune与留出评分均完全相同；单个重复对不等于完整后端噪声分布。ICT单教师在这80股的方向差值区间低于0，但仅一个种子、与主试验共享日期，且没有ICT专属打乱控制，仍不能当作独立语义归因验证。</p>',
                  '<h2>为什么仍弱于先验</h2><p>无教师平均预测概率为48.61% / 6.95% / 44.44%，真实比例约49.49% / 7.64% / 42.86%，总体比例已较接近。其相对先验的LogLoss差额约+0.019913，三类贡献分别为+0.005044 / +0.008278 / +0.006590，平盘类别贡献最大。</p>',
                  '<p>逐种子把全留出期的平均预测向量作为事后常数，LogLoss约0.909136；原模型为0.928067，差额约0.018931。窗口间概率变化没有带来相应区分力，损失主要不再是旧试跑那样的大幅总体类别偏差。这是算术诊断，不是因果证明：全期平均预测在历史原点不可得，未把它拟合成校准器或部署政策。</p>',
                  '<p>12组均在预设检查点中的约一遍训练位置取得最佳tune结果，后半程评分回升，增加更新数没有持续改善方向。教师事实重建目标描述已知历史结构，其改善并不自动带来5根未来日线方向增益。</p>',
                  '<p>固定4窗训练诊断中，双教师最终共享LoRA梯度范数约为AR 2.536、方向0.986、CZSC 0.0337、ICT 0.1157，提示辅助影响相对较小；这是局部诊断。所选LoRA确实已改变：教师相对无教师的有效差异约为基座q/v范数的0.33% / 0.64% / 0.43%，整体适配约5.8%，所以不能说教师从未影响主干，也不能由参数变化直接推出预测价值。</p>',
                  '<p>共同微调把官方条件token NLL的s1/s2约3.30849 / 2.22364降至无教师3.03878 / 2.00920；教师组的两项NLL均略差于无教师。这里使用真实未来前缀做teacher forcing，不能解释为自由生成价格路径或交易收益。</p>',
                  '<h2>执行、来源与证据边界</h2><p>正式12组训练加评估实测约4小时29分，其中优化训练约3小时49分。包含公共缓存、80股前置对照及主报告的study流程约5小时27分，另有数据生成和后续补充评分。合成回归184项通过、56项子测试通过，1项可选依赖检查跳过；两张标准Matplotlib图已查看排版。</p>',
                  '<p>引用源码、日历、官方权重/tokenizer、固定ICT文件及本轮冻结代码共75项哈希匹配。源数据库size/mtime未变，未将stat冒充完整数据库哈希。代码和派生输出保存在本工程；没有训练全市场、修改官方资产、晋升现任模型或拟合交易政策。</p>',
                  '<p>本轮沿用旧研究市场日期，属于回溯验证。历史摄取/修订时点无法从源库还原；未来标签跳空筛选剔除26窗属于事后条件，不能当作预测原点可执行筛选；未确认公司行动原因或进行价格复权。只有24个留出日期，区间条件于三个固定种子且未校正多重比较。当前可以确认内部事实重建改善，尚不能确认稳健预测转移、超越简单先验或双教师互补收益。</p>',
                  '<p><a href="../ablation240_independent_20261021.json">原始预测独立复算</a> · <a href="../ablation240_prediction_variation_20261021.json">概率波动诊断</a> · <a href="../../../audit/20261021_ablation240_population_postcheck.local.json">实际阶段覆盖和时间</a> · <a href="../../../audit/20261021_ablation240_source_postcheck.local.json">来源核验</a> · <a href="../../../logs/ablation240_final_tests_20261021.txt">合成测试日志</a></p></html>'])
    target = OUT / 'review.html'
    if target.exists():
        raise FileExistsError(target)
    target.write_text('\n'.join(parts))
    metadata = {'schema': 'completed_study240_narrative_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                'script_sha256': sha(Path(__file__)), 'inputs': {k: {'path': str(v), 'sha256': sha(v)} for k,v in inputs.items()},
                'independent_main_point_metrics_match': True, 'independent_weighted_point_metrics_match': True,
                'review_sha256': sha(target), 'training_performed': False, 'promoted': False}
    with (OUT / 'review_metadata.json').open('x') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(json.dumps({'review': str(target), 'sha256': sha(target)}))


if __name__ == '__main__':
    main()
