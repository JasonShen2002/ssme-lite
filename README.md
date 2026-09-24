# SSME-Lite

用少量真实标签和多个已有分类器的预测概率，估计这些分类器的 Accuracy、AUC、AUPRC 和 ECE，并给出排名和一份可离线打开的 HTML 报告。

方法依据 Shanmugam et al., *Evaluating multiple models using labeled and unlabeled data*, NeurIPS 2025。官方参考实现是 [divyashan/SSME](https://github.com/divyashan/SSME)。本包默认使用 Scott 尺度带宽，固定运行 100 轮 EM；设置 `early_stopping=True` 可改为阈值停止。它不保证在每个任务上都优于只使用标签的估计。

## 安装

```bash
pip install ssme-lite
```

从本仓库安装：

```bash
pip install -e '.[dev]'
```

迁移学习实验额外需要 `pip install -e '.[transfer]'`。核心估计不依赖 PyTorch。

## 用法

准备一组已经拟合好的分类器。它们要有相同的类别集合，并提供 `predict_proba`。

```python
from ssme_lite import PredictionMatrixGenerator, SSMEEstimator, SemiSupervisedSplit

generator = PredictionMatrixGenerator(models)
scores = generator.fit_transform(X)
splitter = SemiSupervisedSplit(n_labeled=20, n_unlabeled=400, random_state=0)
parts = splitter.indices(len(y))
y_partial = splitter.partial_labels(y, parts)

ssme = SSMEEstimator(max_iter=100, early_stopping=False, random_state=0)
ssme.fit(scores[parts["estimation"]], y_partial, model_names=generator.model_names_)
report = ssme.report(n_draws=100, primary_metric="auc")
print(report.ranking("auc"))
report.save("results/my_report")
```

`fit_transform` 只收集概率，不训练模型。二分类 `scores` 的形状是 `(N, M)`，表示正类概率；多分类是 `(N, M, K)`。`y_partial` 与样本逐行对应，未知标签用 `-1`。每一类至少要有一个可见标签。

`report.save` 写出独立的 `report.html`，以及 `metrics.csv`、`metadata.json` 和 `report_data.json`。页面里的区间是给定已拟合后验之后的潜在标签不确定性，不是总体抽样标准误。

已有概率矩阵也可以不经过分类器对象。NPZ 里放 `scores` 和 `y`，再写一份 JSON，然后运行 `ssme-lite my_config.json`。仓库里的 `configs/` 是合成数据、Gaussian 诊断，以及 CivilComments、MultiNLI 的配置。

## PneumoniaMNIST

这是本仓库的主要实验。七个已发布的 PneumoniaMNIST 分类器在官方 test split 上推理，再用 20 个可见标签和 400 个无标签样本做一次 SSME 估计。

打开这两个文件就能看结果，不必先跑推理：

- [pneumoniamnist.ipynb](pneumoniamnist.ipynb)：实验步骤和这次估计的 AUC 排名
- [pneumoniamnist_report.html](pneumoniamnist_report.html)：同一份交互式报告

重新跑这一次演示：

```bash
python experiments/pneumoniamnist/pneumonia_ssme.py
```

AutoML Vision 的三个权重是 TFLite，需要 `pip install ai-edge-litert`。10 个 seed 的 labeled-only 对比已经保存在 `experiments/pneumoniamnist/results/nl20_nu400_m7/`。在 estimation pool 上，Accuracy 的平均绝对误差从 labeled-only 的 0.058 降到 SSME 的 0.021。模型选择和协议说明在 `experiments/pneumoniamnist/AUDIT.md`，完整写在 `experiments/REPORT.md`。

## 迁移学习

另一条链不使用官方 checkpoint。冻结的 ImageNet ResNet18 提取特征，五个 sklearn 分类头只在官方训练集上拟合，再在 test split 上走同一套 `PredictionMatrixGenerator`、`SemiSupervisedSplit` 和 `SSMEEstimator`。

```bash
pip install -e '.[transfer]'
python experiments/transfer/transfer_ssme.py
```

重复比较是 `python experiments/transfer/transfer_models.py --suite`。任意 ImageFolder 可以用 `experiments/transfer/transfer_resnet.py`。说明见 [experiments/README.md](experiments/README.md)。

## 测试

```bash
python -m pytest -q
```

测试覆盖概率对齐、估计器、划分、报告版式和模型贡献度。发布到 PyPI 之前用它们确认包还能跑。
