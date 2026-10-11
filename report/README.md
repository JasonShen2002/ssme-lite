# 课程论文与图表

- [最终论文](main.pdf)：JMLR 格式，正文 4 页、参考文献 1 页、复现附录 1 页。
- [中文导读](main_zh.md)：按论文结构说明贡献、证据与复现方式。
- [Overleaf 源码包](ssme_lite_report_source.zip)：包含 `main.tex`、附录、文献、模板与矢量图。
- [图表来源](evidence_manifest.json)与[扩展图来源](extension_figure_provenance.json)：保存输入哈希。
- [排版验证](qa/verification.json)：页数、章节分布、最终 PDF 哈希。

## 从数据重建图表

在仓库根目录执行（无需重新训练模型）：

```bash
python report/build_report.py
python report/build_extensions.py
```

全部输入均位于本仓库。第一条命令独立重聚合主实验、生成结果表及三幅图；第二条生成论文使用的四联扩展图。图形同时导出矢量 PDF 与高分辨率 PNG。所有图表中的均值、区间和对照来自已保存结果。

## 编译

将源码包上传 Overleaf，选择 `main.tex` 作为主文件，使用 pdfLaTeX 编译。也可在本目录执行 `tectonic main.tex`。正文沿用 11 pt 字号与 JMLR 页边距；附录包括 OpenI 链接和可执行复现步骤。
