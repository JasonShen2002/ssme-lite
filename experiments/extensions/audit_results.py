"""Independent checks of delivered raw records; no estimator refits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from experiments.extensions.common import ROOT, digest, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'experiments/extensions/results')
    parser.add_argument('--profile',default='full')
    args=parser.parse_args()
    checks={'profile':args.profile,'raw_conditions':0,'checked_statistic_rows':0,
            'checked_accuracy_truth_values':0,'source_snapshot_files':0,'method_missing_class':0}
    for study in ('performance','labels','stress'):
        directory=args.results/study/args.profile
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['complete'] and len(manifest['conditions'])==manifest['expected_conditions']
        for name,expected in manifest['provenance']['files'].items():
            snapshot=directory/'source_snapshot'/name
            source=snapshot if snapshot.is_file() else ROOT/name
            assert digest(source)==expected,(study,name)
            checks['source_snapshot_files']+=int(snapshot.is_file())
        seen_splits={}
        for item in manifest['conditions']:
            path=directory/item['result']
            assert digest(path)==item['sha256']
            r=json.loads(path.read_text());job=r['job']
            assert r['status']=='ok',item
            checks['raw_conditions']+=1
            if study=='performance':
                a=np.array(r['comparison_values'])
                assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
                if job['kind']!='prediction':
                    np.testing.assert_allclose(a.sum(1),1,atol=1e-12)
                    assert r['iterations']==100
                continue
            d=r['diagnostics']
            pool=np.array(d['pool_indices']);heldout=np.array(d['heldout_indices'])
            assert not set(pool)&set(heldout)
            assert d['labeled_indices']==pool[:job['budget']].tolist()
            group=(job.get('pipeline'),job.get('classes'),job['seed'])
            split=(pool.tolist(),heldout.tolist())
            if group in seen_splits:assert seen_splits[group]==split
            seen_splits[group]=split
            src=(ROOT/'reproduction/inputs'/f"{job['pipeline']}.npz" if study=='labels' else
                 ROOT/'experiments/extensions/inputs'/f"digits{job['classes']}.npz")
            with np.load(src,allow_pickle=False) as f:
                p,y,names=f['scores'],f['y'],f['model_names'].tolist()
            if p.ndim==2:p=np.stack([1-p,p],axis=-1)
            if study=='stress':names=names[:3]
            missing=len(np.unique(y[pool[:job['budget']]]))<p.shape[-1]
            assert (r['method_status']['ssme']=='missing_labeled_class')==missing
            checks['method_missing_class']+=int(missing)
            if not missing:assert d['iterations']==100
            for target,idx in [('fixed_pool',pool),('heldout',heldout)]:
                truth=(p[idx,:len(names)].argmax(2)==y[idx,None]).mean(0)
                for entry in r['details']:
                    if entry['target']==target and entry['metric']=='accuracy':
                        assert abs(entry['truth']-truth[names.index(entry['model'])])<1e-12
                        checks['checked_accuracy_truth_values']+=1
            for row in r['rows']:
                entries=[x for x in r['details'] if all(x[k]==row[k] for k in ['method','target','metric'])]
                assert len(entries)==len(names)
                finite=all(x['estimate'] is not None and x['truth'] is not None for x in entries)
                if not finite:
                    assert row['mae'] is None
                    continue
                a=np.array([x['estimate'] for x in entries]);b=np.array([x['truth'] for x in entries])
                assert abs(np.abs(a-b).mean()-row['mae'])<1e-12
                chosen=np.flatnonzero(np.isclose(a,a.max(),rtol=0,atol=1e-12))
                assert abs((b.max()-b[chosen]).mean()-row['selection_regret'])<1e-12
                assert abs(np.isclose(b[chosen],b.max(),rtol=0,atol=1e-12).mean()-row['top1'])<1e-12
                checks['checked_statistic_rows']+=1
    # Preserve train/evaluation separation in the bundled preparation artifacts.
    digits=json.loads((ROOT/'experiments/extensions/inputs/digits_manifest.json').read_text())
    for dataset in digits['datasets']:
        assert not set(dataset['train_indices'])&set(dataset['eval_indices'])
        assert digest(ROOT/'experiments/extensions/inputs'/dataset['file'])==dataset['sha256']
    demo=ROOT/'experiments/extensions/model_cli_demo'
    meta=json.loads((demo/'preparation.json').read_text())
    assert not set(meta['training_indices'])&set(meta['evaluation_indices'])
    receipt=json.loads((demo/'report/input_provenance.json').read_text())
    assert digest(demo/'evaluation.npz')==receipt['data_sha256']
    for m in receipt['models']:assert digest(demo/m['path'])==m['sha256']
    with np.load(demo/'evaluation.npz') as f:assert int((f['y']>=0).sum())==20
    checks.update(status='passed',audit_script_sha256=digest(Path(__file__)))
    write_json(args.results/f'audit_{args.profile}.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
