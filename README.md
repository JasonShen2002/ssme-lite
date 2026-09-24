# SSME-Lite

用少量真实标签和多个已有分类器的预测概率，估计模型性能并辅助选择。
支持二分类/多分类、Accuracy/AUC/AUPRC/ECE、排名、条件标签不确定性图及 HTML/CSV 报告。

## 安装

```bash
pip install ssme-lite
```

从源码安装，并跑测试和快速示例：

```bash
pip install -e '.[dev]'
python examples/quickstart.py
python -m pytest -q
```

用 conda 准备同一套核心依赖：

```bash
conda env create -f environment.yml
conda activate ssme
```

`conda/meta.yaml` 是上传 PyPI 之后再用的 conda-forge 配方。核心运行不需要 torch；迁移学习示例需要 `pip install -e '.[transfer]'`。官方 SSME 参考代码单独克隆自 [divyashan/SSME](https://github.com/divyashan/SSME)，不包含在本仓库和发布包里。

## 已有模型 API

```python
import numpy as np
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator

# models = {"logistic": fitted_lr, "random_forest": fitted_rf, ...}
generator = PredictionMatrixGenerator(models)
scores = generator.fit_transform(X)
# 按 generator.classes_ 的位置编码标签；-1 表示未知。
# y_partial 必须与 X 逐行对应，取值为 -1 或 0..K-1。
ssme = SSMEEstimator(max_iter=100, random_state=0)
ssme.fit(scores, y_partial, model_names=generator.model_names_)
posterior = ssme.predict_proba(scores)
report = ssme.report(metrics=["accuracy", "auc", "auprc", "ece"])
report.show()
print(report.ranking("auc"))
report.save("results/my_report")
```

`fit_transform` 不训练模型；需要先准备已拟合的 sklearn-style 分类器。
各模型类集合须相同，概率列会自动对齐。二分类 scores 为 `(N,M)`，表示 `classes_[1]`
的概率；多分类为 `(N,M,K)`。也可直接输入完整概率张量。输入概率不会被原地修改。
至少每类一个可见标签，否则明确报错；未知类别或 NaN 概率不被静默修复。

另一种输入方式：`ssme.fit(scores_labeled, y_labeled, scores_unlabeled)`。
默认固定运行 100 轮 EM：`max_iter=100, early_stopping=False`。
`max_iter` 在固定模式下就是实际轮数，不因 `tol` 提前停止；CivilComments 作者带宽配置
明确使用 `max_iter=20`。`tol=1e-3` 仍用于记录最终变化是否达到参考阈值。
如需恢复阈值停止，显式设置 `early_stopping=True`，此时 `max_iter` 为迭代上限。
benchmark 同样支持上述参数，并保存 `early_stopping`、实际迭代数及 `stop_reason`。
默认标注权重为 `labeled_weight=10.0`，未标注样本权重为 1。
历史结果不随默认值改变而更新；复现过去的阈值停止实验需显式指定 `early_stopping=True`，
并使用该次保存的 `max_iter`、`tol` 和 `labeled_weight`。
带宽默认仍为 `scott`。复现官方公开仓库的 KDE 带宽可使用：

```bash
pip install -e '.[official]'
ssme-lite configs/civilcomments_official_kde.json
```

API：`SSMEEstimator(bandwidth="official", labeled_weight=10, max_iter=20, early_stopping=False)`。
`official` 使用各 ALR 维度的 statsmodels normal-reference 带宽除以 4，再取最小值。
这是公开仓库的实际规则，不称为论文的 Improved Sheather-Jones。
它仍保留本包的软先验、标注责任度固定和变化量诊断，不是整个官方 EM 的逐行复刻。
默认报告所有拟合样本；只评估未标注部分可用 `report(target="unlabeled")`。
`predict_proba` 为纯模型推断；`posterior_` 才包含拟合池已知标签的 one-hot 固定值。

## 可复现实验

```bash
ssme-lite configs/synthetic.json
ssme-lite configs/gaussian.json
ssme-lite configs/gaussian_1000.json
ssme-lite configs/civilcomments.json
ssme-lite configs/multinli.json
python -m pytest -q
```

Synthetic 包含 10 个不同 sklearn 分类器，训练集与评估集分离。
Gaussian 是另一个受控诊断场景：给定类别时，10 个 score view 来自独立高斯分布，
用于验证方法在适合混合密度建模的设定中是否工作，不代表真实任务效果。
CivilComments 使用官方提供的 7 个模型；MultiNLI 使用 4 个模型，不能把同一模型复制凑到 10 个。
默认每次只暴露 20 标签，使用 1000 个未标注样本。每次保存随机分割，不按表现挑选 seed。
缺少标注类别的分割会记录失败，不通过查看隐藏标签补充。

输出 `results/<dataset>/`：

- `summary.csv`：平均误差、Spearman、Top-1、selection regret，以及跨 seed SEM/t 区间。
- `details.csv` / `per_seed.csv`：逐模型、逐指标、逐次结果。
- `config.json` / `splits.json` / `run_status.json`：版本、参数、数据索引、收敛/失败记录。
- `comparison.png`：与 labeled-only 的 heldout 误差比较。
- `report/report.html`：首个成功分割的用户报告（不是跨 seed 汇总）。
- `contribution.csv`：启用时输出逐个移除模型的敏感度。

## 自己的预测数据与一键报告

NPZ 需包含 `scores`、`y`，可选 Unicode 数组 `model_names`。`y=-1` 表示未标注。
采用 `numpy.savez_compressed` 保存，不接受需 pickle 才能读取的对象数组。

```json
{
  "source": "npz",
  "mode": "evaluate",
  "data": "scores.npz",
  "output": "my_report",
  "estimator": {"max_iter": 100, "early_stopping": false, "random_state": 0},
  "report": {"n_draws": 100, "target": "all"},
  "contribution": false
}
```

保存为 `my_config.json` 后执行 `ssme-lite my_config.json`。
相对路径基于 JSON 文件所在目录。相同输出目录会更新同名报告文件。
完整标签数据也可指定 `mode: "benchmark"`，通过 `benchmark` 字段控制随机分割。

## 交互式 Real-world HTML 报告

`report.save(...)` 现在生成独立、可离线分享的交互式 HTML：首屏性能概览、
多指标矩阵与排名小图、抽样排名热图、两两比较、metric sampling 分布、
模型相关性和一致率、ALR/PCA 预测空间、后验诊断以及逐轮拟合诊断。
顶部指标选择会联动性能、排名、分布和最终摘要；模型按钮可固定跨图高亮，
PCA 支持四种配色与已标注锚点突出显示。页面适配移动端并带打印样式。

```python
report = ssme.report(
    n_draws=500,
    title="CivilComments · Model Evaluation",
    primary_metric="auc",
    target="all",
    # 可选：与完整拟合池逐行对应的原始样本 ID
    # sample_ids=sample_ids,
)
report.save("results/my_report")
```

新增 `report_data.json` 保存可重建图表的结构化统计（包含逐模型 metric samples、
排名频率、配对差值分布、PCA 坐标与拟合记录）；HTML 内嵌同一数据，并提供导出按钮。
已有的 `metrics.csv`、`metadata.json` 与四张 PNG 继续保留。
默认 HTML 不使用完整 ground truth；benchmark 的误差对比仍在报告目录外的
`summary.csv` 和 `comparison.png` 中。HTML 对应一个分割，不是跨 seed 汇总。

排名使用对全部候选模型均有效的抽样，并列时平分其占据的排名位置；
pairwise frequency 使用对当前模型对均有效的抽样，明确区分胜出、落后与并列。
ECE 自动使用 lower-is-better 方向。Rank-1 stability 是潜在标签抽样频率，
不能解释成模型真实最优的概率。

后验诊断只统计未标注样本；PCA 和模型关系使用完整拟合池。
大于 3000 个点时，PCA 保留全部已标注锚点，对未标注点确定性抽样展示，
所有汇总统计仍基于全量数据。默认不进行额外的 robustness 重复拟合。
单次 EM 迭代显示单点，metric 抽样分布无可见变化时也显示单点。

`examples/refresh_reports.py` 可利用已有 `report_data.json` 刷新版式而不重跑实验；
只有旧版 CSV/metadata 的目录仍使用旧版简报，需要重新拟合才能获得新增诊断。

### 指标和不确定性口径

Accuracy 用后验精确期望；AUC/AP/ECE 用多次潜在标签抽样。已标注样本始终使用真值。
ECE：二分类校准正类概率，多分类校准 top-label，采用 10 个等宽区间、样本数加权。
多分类 AUC 和 AP 采用 macro OvR。抽样未覆盖所有类别时 AUC/AP 返回 NaN，报告有效抽样次数。
ECE 越低越好，其他指标越高越好；并列排名使用相同名次。

图中的区间只反映**给定已拟合后验的潜在标签不确定性**，不包含密度估计、
带宽选择或总体抽样误差。`mc_standard_error` 是数值 Monte Carlo 误差，
不能当作真实性能估计的标准误。跨 seed 实验区间另行保存在 `summary.csv`。

## 模型贡献度与迁移学习

`ssme.report(contribution=True)` 在 Model Landscape 后增加贡献度章节，
执行 M 次受控的 leave-one-model-out 重新拟合。`ssme.contribution()` 可直接返回明细表。
CivilComments 作者带宽配置已启用，其他报告默认不进行这组额外拟合。

受控条件：共享全模型的初始 responsibilities、原始可见标签、标量带宽，以及实际 EM 轮数。
所有影响只统计未标注样本，包括平均后验 total variation、JS divergence、后验类别翻转率，
剩余模型的 Accuracy 估计变化及名次变化。Accuracy 使用精确后验期望，没有标签抽样噪声；
排名仅在仍保留的候选模型之间比较。热图显示“移除后 − 移除前”的估计变化，单位为百分点。

这衡量固定初始化下的条件影响，不是准确性增益或 Shapley value。
全模型初始化本身包含所有模型的信息，该控制诊断不衡量移除模型对初始化的影响；
也不把重新选带宽的变化混入逐模型影响。低影响可能来自预测冗余或后验集中，不能据此断言模型无用。
结果进入 HTML、`report_data.json` 和 `contribution.csv`，benchmark 另保存各 seed 的明细 CSV。

报告将 Ranking Stability 和 Metric Sampling 合并为 Sampling & Ranking。
Prediction Space 的切换只改变同一 PCA 图上的颜色：推断类别、最大后验概率、
后验熵或是否有已知标签；每种模式都显示解释和数值范围。

课程实验在 `experiments/`，说明见 [experiments/README.md](experiments/README.md)。

```bash
python experiments/pneumoniamnist/pneumonia_ssme.py
pip install -e '.[transfer]'
python experiments/transfer/transfer_ssme.py
```

`examples/transfer_resnet.py` 提供任意 ImageFolder 的入口：冻结 ImageNet ResNet18 → 特征 → sklearn 分类器 → SSME。`examples/transfer_learning.ipynb` 走同一条 PneumoniaMNIST 迁移链。协议和审计见 `docs/PNEUMONIAMNIST_AUDIT.md`。

```bash
pip install -e '.[transfer]'
python examples/transfer_resnet.py --images /path/to/imagefolder --output results/transfer
```

图像按类分子目录；需要足够图像以分离分类器训练集、1020 样本评估池及 heldout。
首次运行下载公开预训练权重；图像留在本机。未提供图像路径时，这条通用入口本身不算已经跑完迁移实验。PneumoniaMNIST 示例跑完后，以 `experiments/transfer/results/` 为准。

## 算法依据与边界

方法依据 Shanmugam et al., *Evaluating multiple models using labeled and unlabeled data*, NeurIPS 2025。
官方参考实现是 [divyashan/SSME](https://github.com/divyashan/SSME)。
本实现默认使用 Scott 尺度带宽，可选公开仓库的 official 规则，未实现论文 improved Sheather-Jones。
估计器默认固定运行 100 轮，设置 `early_stopping=True` 可恢复阈值停止。
不会保证每个任务/指标都优于 labeled-only。PneumoniaMNIST 的模型审计和课程实验写在 `docs/`。
