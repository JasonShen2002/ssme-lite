# PneumoniaMNIST 数据与模型审计

结论：**通过**。正式实验使用官方 test split 上、预先指定的 5 个不同族模型。

## 数据

PneumoniaMNIST 是儿科胸片的二分类（0 = normal，1 = pneumonia）。官方划分是 train 4708 / val 524 / test 624。本审计的 test 类别计数为 normal 234、pneumonia 390。

分类器在 train 上训练，并用 val 做 early stopping。SSME 的评估池因此只用 **test**。test 只有 624 张，放不下论文里的 1,000 个无标签样本。协议改为 `n_l ∈ [20, 50, 100]`、`n_u = 400`，并保留剩余样本作 held-out。`n_l = 100` 时 held-out 为 124。`n_u >> n_l` 在 20 和 50 个标签时成立，在 100 个标签时是 4 倍。

验证集不并入评估池。迁移学习示例另行在 train 上拟合 sklearn 头，也只在 test 上评估；ImageNet backbone 没有在这份数据上训练。

## 官方文件里有什么

来源是 Zenodo `10.5281/zenodo.7782114` 的 `predictions.zip`。归档约 1.9 GB，这里只取 PneumoniaMNIST 的 test CSV。每个文件由 `medmnist.Evaluator` 写出，二分类分数用 0.5 阈值算 accuracy、用该分数算 ROC-AUC，因此是概率（或与概率同一量纲的分数），不是 hard label。
auto-sklearn 的模型快照没有发布，但它的预测文件在。SSME 消费的是预测概率，不需要把 853 MB 权重再跑一遍。

test 预测共 21 个文件：ResNet-18/50 各有 28 与 224 两种输入，再加 AutoKeras、auto-sklearn、Google AutoML Vision，每个族 3 次重复。

## 选定的模型

规则在跑 SSME 之前写定：每个族只取重复编号 `1`，不按 SSME 误差挑 checkpoint。
四个 ResNet 设置不再各占一个名额；28 像素的 ResNet-18 和 224 像素的 ResNet-50 保留深度和输入尺寸两种差异，其余名额留给不同训练系统。

| 模型 | 输入（文件名） | 输出 | 文件 AUC | 重算 AUC | 文件 ACC | 重算 ACC | 分数范围 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `resnet18_28_1` | ResNet-18 @ 28 | 2 列类别概率，取正类 | 0.948 | 0.948 | 0.833 | 0.833 | [0.000, 1.000] |
| `resnet50_224_1` | ResNet-50 @ 224 | 2 列类别概率，取正类 | 0.967 | 0.967 | 0.896 | 0.896 | [0.000, 1.000] |
| `autokeras_1` | AutoKeras | 1 列正类概率 | 0.939 | 0.939 | 0.883 | 0.883 | [0.000, 1.000] |
| `autosklearn_1` | auto-sklearn | 2 列类别概率，取正类 | 0.942 | 0.942 | 0.859 | 0.859 | [0.377, 0.629] |
| `automl_vision_1` | Google AutoML Vision | 2 列类别概率，取正类 | 0.993 | 0.993 | 0.941 | 0.941 | [0.039, 0.969] |

Accuracy 极差 0.107。阈值预测的最大不一致率 0.120。auto-sklearn 的正类概率挤在 0.377–0.629：排序仍有效（AUC 0.942），置信度明显更低，这是校准差异，文件本身和重算指标一致。

正类概率的 Pearson 相关：

| | `resnet18_28_1` | `resnet50_224_1` | `autokeras_1` | `autosklearn_1` | `automl_vision_1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_28_1` | 1.000 | 0.829 | 0.881 | 0.868 | 0.804 |
| `resnet50_224_1` | 0.829 | 1.000 | 0.912 | 0.892 | 0.894 |
| `autokeras_1` | 0.881 | 0.912 | 1.000 | 0.928 | 0.887 |
| `autosklearn_1` | 0.868 | 0.892 | 0.928 | 1.000 | 0.891 |
| `automl_vision_1` | 0.804 | 0.894 | 0.887 | 0.891 | 1.000 |

0.5 阈值预测不一致率：

- `resnet18_28_1|resnet50_224_1`: 0.104
- `resnet18_28_1|autokeras_1`: 0.082
- `resnet18_28_1|autosklearn_1`: 0.067
- `resnet18_28_1|automl_vision_1`: 0.120
- `resnet50_224_1|autokeras_1`: 0.064
- `resnet50_224_1|autosklearn_1`: 0.085
- `resnet50_224_1|automl_vision_1`: 0.074
- `autokeras_1|autosklearn_1`: 0.056
- `autokeras_1|automl_vision_1`: 0.090
- `autosklearn_1|automl_vision_1`: 0.101

## 模型数量消融的嵌套子集

子集按预测相关从低到高往上加，只用分数、不用标签，也不看 SSME 误差。主实验的 5 模型顺序仍是上表顺序。

- 2 个模型: `resnet18_28_1`, `automl_vision_1`
- 3 个模型: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`
- 4 个模型: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`, `autosklearn_1`
- 5 个模型: `resnet18_28_1`, `automl_vision_1`, `resnet50_224_1`, `autosklearn_1`, `autokeras_1`

同一 seed 的划分只依赖样本数，因此这些子集比较的是同一批 labeled / unlabeled / held-out 索引。

