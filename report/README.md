# Paper and figures

- [Final paper](main.pdf): JMLR format, four body pages, one reference page, and one reproduction appendix.
- [Project overview](overview.md): contributions, experimental evidence, and reproduction entry points.
- [Overleaf source archive](ssme_lite_report_source.zip): main document, appendix, bibliography, template, and vector figures.
- [Main figure provenance](evidence_manifest.json) and [extension figure provenance](extension_figure_provenance.json): input hashes.
- [Layout verification](qa/verification.json): page count, section placement, and PDF hash.

## Rebuild figures

Run from the repository root; no model training is required:

```bash
python report/build_report.py
python report/build_extensions.py
```

All inputs are included in this repository. The first command reaggregates the main experiments and generates the results table and figures. The second generates the four-panel experimental figure. Figures are exported as vector PDF and high-resolution PNG; means, intervals, and baselines come from saved results.

## Compile

Upload the source archive to Overleaf, select `main.tex`, and compile with pdfLaTeX. Alternatively, run `tectonic main.tex` in this directory. The paper retains 11 pt text and JMLR margins. The appendix includes the OpenI project and executable reproduction commands.
