"""Summarize saved evidence without refitting, with paired comparisons and audits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.extensions.common import ROOT, digest, paired_statistics, write_json


def read_runs(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    runs = []
    for item in manifest['conditions']:
        path = directory/item['result']
        if digest(path) != item['sha256']:
            raise ValueError(f'result hash mismatch: {path}')
        runs.append(json.loads(path.read_text()))
    return manifest, runs


def audit_splits(runs):
    errors, previous = [], {}
    for run in runs:
        if run['status'] != 'ok':
            continue
        job, d = run['job'], run['diagnostics']
        pool, heldout, labeled = d['pool_indices'], d['heldout_indices'], d['labeled_indices']
        if set(pool)&set(heldout) or labeled != pool[:job['budget']] or len(set(pool+heldout)) != len(pool+heldout):
            errors.append(job)
        key = (job.get('pipeline'), job.get('classes'), job['seed'])
        if key in previous and previous[key] != (pool, heldout):
            errors.append(job)
        previous[key] = (pool, heldout)
    if errors:
        raise ValueError(f'split isolation/nesting audit failed: {errors[:2]}')
    return dict(disjoint_pool_and_heldout=True, nested_labels=True, fixed_target_across_budgets=True,
                checked_conditions=sum(r['status'] == 'ok' for r in runs))


def statistical_summary(directory, study):
    manifest, runs = read_runs(directory)
    audit = audit_splits(runs)
    rows, details, states, diagnostics = [], [], [], []
    for run in runs:
        job = run['job']
        if run['status'] != 'ok':
            states.append(dict(**job, method='all', status=run['status']))
            continue
        rows += [dict(**job, **r) for r in run['rows']]
        details += [dict(**job, **r) for r in run['details']]
        states += [dict(**job, method=m, status=s) for m, s in run['method_status'].items()]
        diagnostics.append({**job, **{k:v for k,v in run['diagnostics'].items() if not isinstance(v, (dict,list))}})
    pd.DataFrame(states).to_csv(directory/'status.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(directory/'diagnostics.csv', index=False)
    pd.DataFrame(details).to_csv(directory/'details.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(directory/'per_seed.csv', index=False)
    if table.empty:
        return
    groups = (['pipeline'] if study == 'labels' else ['classes','regime']) + ['budget','target','metric','method']
    summary = table.groupby(groups, dropna=False).agg(mae=('mae','mean'), mae_sd=('mae','std'),
        valid_seeds=('mae','count'), selection_regret=('selection_regret','mean'), top1=('top1','mean')).reset_index()
    summary.to_csv(directory/'summary.csv', index=False)
    write_json(directory/'audit.json', dict(**audit, profile=manifest['profile'],
               complete=manifest.get('complete', False), expected_conditions=manifest['expected_conditions']))
    if study == 'labels':
        paired = []
        primary = table[(table.target=='heldout') & (table.metric=='accuracy')]
        for pipeline in ('checkpoints','transfer'):
            sub = primary[primary.pipeline==pipeline]
            a = sub[(sub.method=='ssme') & (sub.budget==20)].set_index('seed').mae
            for method,budget in [('labeled_only',50),('labeled_only',100),('mean_probability',20)]:
                b = sub[(sub.method==method) & (sub.budget==budget)].set_index('seed').mae
                common = pd.concat([a.rename('ssme20'),b.rename('comparator')],axis=1).dropna()
                stat = paired_statistics(common.ssme20-common.comparator, family=6)
                # Pilot is not confirmatory evidence.
                if manifest['profile']=='pilot':
                    stat['supports_lower_error'] = False
                paired.append(dict(pipeline=pipeline, comparator=f'{method}{budget}',
                                   family_size=6, profile=manifest['profile'], **stat))
        pd.DataFrame(paired).to_csv(directory/'paired_comparisons.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
        for ax,pipeline in zip(axes,('checkpoints','transfer')):
            sub = summary[(summary.pipeline==pipeline)&(summary.target=='heldout')]
            for method in ('labeled_only','mean_probability','ssme'):
                t = sub[sub.method==method].sort_values('budget')
                ax.plot(t.budget,100*t.mae,'o-',label=method)
            ax.set(title=pipeline,xlabel='Visible labels',ylabel='Held-out Accuracy MAE (pp)')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(directory/'label_efficiency.png',dpi=170)
        plt.close(fig)
    else:
        pairs = []
        # Compare regimes on shared successful seeds only; target/candidates fixed.
        for k in (3,10):
            for budget in (20,60):
                sub=table[(table.classes==k)&(table.budget==budget)&(table.method=='ssme')&
                          (table.target=='fixed_pool')&(table.metric=='accuracy')]
                pivot=sub.pivot(index='seed',columns='regime',values='mae')
                for other in ('six_diverse','three_duplicated'):
                    if 'three_diverse' in pivot and other in pivot:
                        dif=(pivot[other]-pivot.three_diverse).dropna()
                        pairs.append(dict(classes=k,budget=budget,comparison=other+' minus three_diverse',
                                          n=len(dif),mean_mae_change=float(dif.mean()) if len(dif) else None))
        pd.DataFrame(pairs).to_csv(directory/'regime_pairs.csv', index=False)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,k in zip(axes,(3,10)):
            sub=summary[(summary.classes==k)&(summary.budget==20)&(summary.target=='fixed_pool')&
                        (summary.metric=='accuracy')]
            regimes=['three_diverse','six_diverse','three_duplicated']
            for i,method in enumerate(('labeled_only','mean_probability','ssme')):
                t=sub[sub.method==method].set_index('regime').reindex(regimes)
                ax.bar(np.arange(3)+(i-1)*.24,100*t.mae,width=.24,label=method)
            ax.set_xticks(np.arange(3),['3 diverse','6 diverse','3 duplicated'])
            ax.set(title=f'{k} classes; 20 labels',ylabel='Fixed-pool Accuracy MAE (pp)')
            ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(directory/'stress.png',dpi=170)
        plt.close(fig)


def performance_summary(directory):
    manifest,runs=read_runs(directory)
    rows=[]
    groups={}
    for r in runs:
        job=r['job']
        rows.append({**job,**{k:v for k,v in r.items() if k not in ('job','comparison_values','traceback')}})
        if r['status']=='ok':
            key=(job['kind'],job['n'],job['models'],job['classes'],job['iterations'],r['input_sha256'])
            groups.setdefault(key,[]).append(r)
    table=pd.DataFrame(rows)
    table.to_csv(directory/'per_run.csv',index=False)
    checks=[]
    for key,items in groups.items():
        reference=next((r for r in items if r['job']['workers']==1 and r['job']['batch']==512),items[0])
        for r in items:
            delta=float(np.max(np.abs(np.asarray(r['comparison_values'])-reference['comparison_values'])))
            checks.append(dict(kind=key[0],n=key[1],models=key[2],classes=key[3],workers=r['job']['workers'],
                               batch=r['job']['batch'],repetition=r['job']['repetition'],max_abs_difference=delta,
                               passed=delta<=1e-12))
    pd.DataFrame(checks).to_csv(directory/'equivalence.csv',index=False)
    if not all(c['passed'] for c in checks):
        raise ValueError('parallel/batch equivalence check failed')
    group=['kind','n','models','classes','workers','batch','iterations']
    cols=[c for c in ['fit_seconds','metric_seconds','inference_seconds','report_seconds','prediction_seconds',
                     'peak_rss_mib','report_bytes'] if c in table]
    ok=table[table.status=='ok']
    summary=ok.groupby(group)[cols].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(directory/'summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    t=ok[(ok.kind=='controlled')&(ok.models==3)&(ok.classes==2)&(ok.workers==1)&(ok.batch==512)]
    a=t.groupby('n').agg(time=('fit_seconds','mean'),rss=('peak_rss_mib','mean'))
    axes[0].plot(a.index,a.time,'o-');axes[0].set(xlabel='Samples',ylabel='Fit time (s)')
    axes[1].plot(a.index,a.rss,'o-');axes[1].set(xlabel='Samples',ylabel='Whole-worker peak RSS (MiB)')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(directory/'scaling.png',dpi=170);plt.close(fig)
    write_json(directory/'audit.json',dict(profile=manifest['profile'],complete=manifest.get('complete',False),
               successful_conditions=len(ok),expected_conditions=manifest['expected_conditions'],
               max_parallel_batch_difference=max((c['max_abs_difference'] for c in checks),default=None)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'experiments/extensions/results')
    parser.add_argument('--profile',choices=['pilot','full','server'],default='full')
    args=parser.parse_args()
    for study in ('performance','labels','stress'):
        directory=args.results/study/args.profile
        if not (directory/'manifest.json').exists():continue
        if study=='performance':performance_summary(directory)
        else:statistical_summary(directory,study)
        print(directory,flush=True)


if __name__=='__main__':main()
