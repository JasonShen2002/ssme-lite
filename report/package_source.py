"""Package the exact, locally compiled Overleaf source without build caches."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
files = [HERE / n for n in ['main.tex','appendix_reproduction.tex','references.bib',
                           'jmlr2e.sty','results_table.tex']]
files += [HERE/'figures'/f'{name}.pdf' for name in ['workflow','diagnostics','extensions']]
output = HERE / 'ssme_lite_report_source.zip'
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(HERE))
    archive.writestr('OVERLEAF.txt', 'Main document: main.tex\nCompiler: pdfLaTeX\nBody: 4 pages; references: 1 page; reproduction appendix: 1 page.\nSource and evidence: https://openi.pcl.ac.cn/jasonshen/SSME-Lite-2026211240\n')
manifest = {'archive_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'files':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(HERE/'source_package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(output.name, output.stat().st_size, 'bytes')
