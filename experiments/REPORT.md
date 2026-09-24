# SSME-Lite 实验报告

PneumoniaMNIST 上的半监督模型评估。比较对象是「只用少量标签」和 SSME-Lite。迁移学习是另一条独立的链：冻结的 ImageNet ResNet18 提特征，再接 sklearn 分类头。

## 1. 问题

Accuracy、AUC、AUPRC、ECE 通常要在一大批有标签的样本上算。儿科胸片这类任务标签少，未标注图像反而多。SSME（Shanmugam et al., NeurIPS 2025）用多个分类器的预测概率，加上少量真实标签，估计这些分类器在未标注数据上的性能。

SSME-Lite 把这件事做成可调用的流程，而不是再改一版核心算法：

`多个模型的概率 → PredictionMatrixGenerator → SSMEEstimator → 排名、区间和报告`

本报告回答三件事。官方 PneumoniaMNIST 模型能不能组成一个有差异的分类器集合。在统一划分上，SSME-Lite 的绝对误差是否低于只使用标签的估计。预训练 ResNet 提取的特征能不能接进同一条流程。

## 2. 方法

估计器沿用工具包里已经实现的流程：把多个模型的类别概率做加性对数比变换，在这个空间里做加权高斯核密度估计，再用 EM 更新未标注样本的类别责任度。已见标签在每次 E-step 固定为 one-hot。默认固定跑 100 轮，标注样本权重 10，带宽用 Scott 尺度。Accuracy 用后验期望；AUC、AUPRC、ECE 用潜在标签抽样。这些选择和本仓库其余 benchmark 一致，不是论文 improved Sheather-Jones 带宽的逐行复现。

数据划分封装为 `SemiSupervisedSplit`。一次均匀随机置换把评估池切成三段：

- 模型训练数据：分类器已经用过，SSME 不再使用
- estimation set：`n_l` 个可见标签 + `n_u` 个只提供预测的样本
- held-out：标签完全不进入 SSME，用来计算 ground truth

同一个 seed 只由样本数决定，因此模型数量消融比较的是同一批索引。误差是

\[
|\widehat{\mathrm{Metric}} - \mathrm{Metric}_{GT}|
\]

跨 10 个 seed 取均值，并用 Student-t 区间描述这个均值的不确定性。区间只覆盖随机划分，不覆盖密度估计本身的不确定性。

## 3. 实验设置

### 3.1 数据与官方模型

PneumoniaMNIST 是二分类胸片：train 4,708、val 524、test 624。test 里 normal 234、pneumonia 390。官方分类器在 train 上训练，并用 val 做 early stopping，所以评估池只用 test。624 张放不下论文中的 1,000 个无标签样本，协议定为

\[
n_l \in \{20, 50, 100\}, \quad n_u = 400
\]

`n_l = 20` 时 held-out 为 204；`n_l = 100` 时为 124。

模型来自 MedMNIST v2 发布的 test 预测（Zenodo `10.5281/zenodo.7782114`），不是把同一个 checkpoint 复制五份。每个族只取重复编号 1。重算的 AUC / Accuracy 与文件名里的三位小数一致，分数落在 `[0, 1]`。

| 模型 | 结构 | Test AUC | Test ACC |
| --- | --- | ---: | ---: |
| `resnet18_28_1` | ResNet-18，28×28 | 0.948 | 0.833 |
| `resnet50_224_1` | ResNet-50，224×224 | 0.967 | 0.896 |
| `autokeras_1` | AutoKeras | 0.939 | 0.883 |
| `autosklearn_1` | auto-sklearn | 0.942 | 0.859 |
| `automl_vision_1` | Google AutoML Vision | 0.993 | 0.941 |

Accuracy 极差 0.107。两两阈值预测的不一致率最高 0.120，正类概率相关在 0.80–0.93，不是同一份分数。auto-sklearn 的概率挤在 0.38–0.63，排序还在，置信度更低。它的模型快照没有发布，SSME 用的是发布出来的概率。完整审计在 `docs/PNEUMONIAMNIST_AUDIT.md`。

### 3.2 迁移学习

这条线不加载上面的 checkpoint。ImageNet 预训练 ResNet18 去掉分类层并冻结，28×28 灰度图复制成 3 通道、双线性放大到 224、按 ImageNet 均值方差归一化，得到 512 维特征。Logistic Regression、Random Forest、MLP、Extra Trees、HistGradientBoosting 只在官方 train split 上拟合，超参数事先写定，验证集不参与挑选。SSME 仍只看 test split，划分协议与上一节相同。

### 3.3 重复实验

每种设置 10 个 seed（0–9）。主比较是 5 个模型、`n_l = 20`、`n_u = 400`。然后只改变标签数，或只改变模型个数。2/3/4 个官方模型的子集按预测相关从低到高嵌套加入，规则只用分数，不看 SSME 误差。迁移学习的子集按分类头类型事先指定：logistic + forest，再依次加入 MLP、Extra Trees、HistGradientBoosting。

主实验打开了逐个移除模型的敏感度分析。全部 seed 都跑完 100 轮，没有因标签缺类而丢弃的划分。

## 4. 结果与工具包

### 4.1 官方模型：20 个标签

Held-out 上的平均绝对误差。括号里是 10 个 seed 均值的 95% t 区间。

| 指标 | 只用 20 个标签 | SSME-Lite |
| --- | ---: | ---: |
| Accuracy | 0.053 [0.040, 0.065] | 0.030 [0.023, 0.037] |
| AUC | 0.032 [0.023, 0.041] | 0.030 [0.022, 0.038] |
| AUPRC | 0.033 [0.021, 0.045] | 0.030 [0.021, 0.038] |
| ECE | 0.038 [0.030, 0.045] | 0.033 [0.024, 0.043] |

Accuracy 的两个区间没有重叠，SSME-Lite 更接近 held-out 真值。AUC、AUPRC、ECE 的绝对误差接近，区间重叠。

排序是另一回事。Accuracy 的 Spearman 从 0.71 到 0.78。AUC 的 Spearman 从 0.40 降到约 0，Top-1 从 0.56 降到 0：十次划分里 SSME 都没有把 held-out AUC 最高的模型排到第一。原因在估计值本身。五个模型的真实 estimation-pool AUC 大约从 0.94 到 0.99，SSME 把它们都估到 0.966–0.978。最强的 AutoML Vision（真值约 0.992）被估成 0.966，弱于 ResNet-50。绝对误差不大，名次却反了。同一结论在 estimation pool 的 ground truth 上同样出现，不是 held-out 和估计池不一致造成的。

### 4.2 标签数和模型数

Held-out Accuracy 的平均绝对误差：

| 设置 | 只用标签 | SSME-Lite |
| --- | ---: | ---: |
| `n_l = 20`，5 个模型 | 0.053 | 0.030 |
| `n_l = 50`，5 个模型 | 0.029 | 0.027 |
| `n_l = 100`，5 个模型 | 0.028 | 0.032 |
| `n_l = 20`，2 / 3 / 4 / 5 个模型 | 0.056 / 0.055 / 0.052 / 0.053 | 0.028 / 0.029 / 0.028 / 0.030 |

标签从 20 增到 50 以后，只用标签的 Accuracy 误差降到和 SSME 同一水平。到 100 个标签时，只用标签略好，两者区间重叠。SSME 的 Accuracy 误差从 2 个模型到 5 个模型几乎不变。AUC 的 Top-1 在 3、4、5 个模型时仍是 0。多一个相关的分类器，没有自动带来更好的 AUC 排名。

逐个移除模型（10 个 seed 的平均后验 total variation）显示信息并不均匀：ResNet-50 为 0.041，AutoKeras 和 AutoML Vision 约为 0.020，ResNet-18 为 0.013，auto-sklearn 为 0.0001。auto-sklearn 的概率挤在 0.5 附近，对后验几乎没有推动。这是冗余线索，不是“这个模型没有预测能力”：它的 test AUC 仍有 0.942。

### 4.3 迁移学习

五个 sklearn 头的 test 预测彼此不同。一次划分上，SSME 给出的 AUC 估计从 Extra Trees 的 0.950 到 MLP 的 0.982。重复实验里，`n_l = 20`、5 个分类头的 held-out 绝对误差：

| 指标 | 只用 20 个标签 | SSME-Lite |
| --- | ---: | ---: |
| Accuracy | 0.071 [0.046, 0.096] | 0.042 [0.025, 0.058] |
| AUC | 0.048 [0.033, 0.062] | 0.039 [0.022, 0.056] |
| AUPRC | 0.043 [0.032, 0.053] | 0.040 [0.018, 0.063] |
| ECE | 0.064 [0.042, 0.086] | 0.040 [0.026, 0.053] |

这里 Accuracy 的排序也分开了：Spearman 0.30 对 0.87，Top-1 0.25 对 0.70。ECE 的 Spearman 从 −0.46 到 0.53。AUC 两边都排不好，因为这些头的 AUC 都挤在 0.95 以上。`n_l = 50` 时 Accuracy 误差仍是 0.056 对 0.032；`n_l = 100` 时是 0.043 对 0.034。

移除一个分类头时，MLP 和 Logistic Regression 的后验变化最大（total variation 0.047 和 0.043），两棵树模型只有 0.004 和 0.003。同一组 ResNet 特征上，树模型彼此更替补，线性模型和 MLP 提供的信息更多。

28×28 放大到 224 不会恢复原始胸片分辨率。这个例子证明的是流程能跑通，并且在 20 个标签时 Accuracy / ECE 的估计优于只使用标签，不是肺炎检测的新结果。

### 4.4 怎样复现

```bash
conda activate ssme
python experiments/pneumoniamnist/pneumonia_ssme.py
python experiments/transfer/transfer_ssme.py
```

根目录的 `pneumoniamnist.ipynb` 是同一次官方模型演示。迁移学习的单次演示是 `experiments/transfer/transfer_ssme.py`。跨 seed 的表在 `experiments/pneumoniamnist/results/` 和 `experiments/transfer/results/`。主配置也可以用 `ssme-lite configs/pneumoniamnist.json`。

特征提取用了本机的 PyTorch 与 MPS，ImageNet ResNet18 权重只下载一次。官方实验只读取预测 CSV，不需要 853 MB 的 MedMNIST 权重。

### 4.5 边界

10 个 seed 用来看方向，比论文里常见的 30–50 次少，区间因此更宽。`n_u = 400` 是 test split 的规模决定的，不是 1,000。SSME 在这份数据上改善的是少量标签时的 Accuracy 估计，以及迁移学习里的 Accuracy / ECE；它没有稳定地排出 AUC 名次。模型彼此很强、AUC 很接近时，绝对误差变小并不足以选对模型。
