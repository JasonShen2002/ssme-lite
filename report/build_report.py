"""Build manuscript tables, vector figures and provenance from saved experiments.
Run from any working directory; no model training or SSME fits are performed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = (HERE.parent / 'experiments' if (HERE.parent / 'experiments').is_dir()
       else ROOT / 'ssme-lite/experiments')
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
inputs = []
BLUE, GRAY, GOLD, PURPLE = '#087F8C', '#738296', '#D7A34B', '#81789C'
plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10,
                     'axes.labelsize': 9.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.family': 'DejaVu Sans',
                     'pdf.fonttype': 42, 'axes.titlepad': 8,
                     'legend.fontsize': 9, 'xtick.labelsize': 9,
                     'ytick.labelsize': 9})

def read(path):
    inputs.append(path)
    return pd.read_csv(path)

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=200)
    plt.close(fig)

def mean_ci(values):
    values = pd.Series(values).dropna()
    return values.mean(), t.ppf(.975, len(values)-1) * values.sem()

settings = [('Released models (7)', 'pneumoniamnist/results/nl20_nu400_m7'),
            ('Transfer heads (5)', 'transfer/results/nl20_nu400_m5')]
frames = {}
rows = []
for title, sub in settings:
    d = read(EXP / sub / 'per_seed.csv')
    d = d[d.target == 'heldout']
    assert d.seed.nunique() == 10
    frames[title] = d
    # Independent check: reconstruct split MAE from the model-level evidence.
    details = read(EXP / sub / 'details.csv')
    recomputed = details[details.target == 'heldout'].groupby(
        ['seed', 'method', 'metric']).absolute_error.mean().sort_index()
    recorded = d.set_index(['seed', 'method', 'metric']).mae.sort_index()
    np.testing.assert_allclose(recomputed, recorded, atol=1e-14)
    for metric in ['accuracy', 'auc', 'auprc', 'ece']:
        pair = d[d.metric == metric].pivot(index='seed', columns='method', values='mae') * 100
        delta = pair.labeled_only - pair.ssme
        mu, half = mean_ci(delta)
        rows.append(dict(experiment=title, metric=metric, baseline=pair.labeled_only.mean(),
                         ssme=pair.ssme.mean(), reduction=mu, low=mu-half, high=mu+half, n=len(pair)))
statistics = pd.DataFrame(rows)
statistics.to_csv(HERE / 'report_statistics.csv', index=False)
lines = [r'\renewcommand{\arraystretch}{1.08}',
         r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccc@{}}', r'\toprule',
         r' & \multicolumn{2}{c}{MAE (pp)} & \multicolumn{2}{c}{Paired reduction (pp)} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Metric & Labels only & SSME-Lite & Mean & 95\% CI \\', r'\midrule']
for i, r in enumerate(rows):
    if i % 4 == 0:
        if i: lines.append(r'\midrule')
        group = 'Pretrained checkpoints (7 models)' if i == 0 else 'Transfer learning (5 heads)'
        lines.append(r'\multicolumn{5}{c}{\textit{' + group + r'}} \\')
    metric = dict(accuracy='Accuracy', auc='AUC', auprc='AP', ece='ECE')[r['metric']]
    lines.append(f"{metric} & {r['baseline']:.2f} & {r['ssme']:.2f} & "
                 f"{r['reduction']:.2f} & [{r['low']:.2f}, {r['high']:.2f}] " + r'\\')
lines += [r'\bottomrule', r'\end{tabular*}']
(HERE / 'results_table.tex').write_text('\n'.join(lines))

# Figure 1: software workflow. Diagram text is at its final print size.
fig, ax = plt.subplots(figsize=(6, 1.0))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
boxes = [('Predict', 'Images to scores\nor stored scores'),
         ('Validate', 'Align classes\nand sample rows'),
         ('Infer', 'ALR + weighted KDE\nclamp known labels'),
         ('Report', 'Ranks and metrics\nposterior diagnostics')]
for j, (head, body) in enumerate(boxes):
    x = .008 + j * .255
    ax.add_patch(FancyBboxPatch((x, .24), .220, .67, boxstyle='round,pad=0.005,rounding_size=.025',
                               facecolor='#EDF5F6' if j == 2 else '#F5F6F8',
                               edgecolor=BLUE if j == 2 else '#AEB9C5', linewidth=.8))
    ax.text(x+.1125, .745, head, ha='center', va='center', fontsize=10, weight='bold', color='#22394C')
    ax.text(x+.1125, .465, body, ha='center', va='center', fontsize=9, linespacing=1.3)
    if j < 3:
        ax.add_patch(FancyArrowPatch((x+.232, .57), (x+.25, .57), arrowstyle='-|>', mutation_scale=10,
                                    color=GRAY, linewidth=.9))
ax.text(.5, .075, 'Hidden holdout labels are used only for benchmark scoring.',
        fontsize=9, ha='center', va='center', color='#506171')
save(fig, 'workflow')

# Figure 2: all-model Accuracy estimation errors in the main experiment.
fig, axes = plt.subplots(1, 2, figsize=(6, 2.0))
fig.subplots_adjust(left=.16, right=.98, bottom=.23, top=.74, wspace=.93)
model_rows = []
model_groups = [
    (['resnet18_28_1', 'autokeras_1', 'autokeras_2', 'autokeras_3',
      'automl_vision_1', 'automl_vision_2', 'automl_vision_3'],
     ['ResNet18', 'AutoKeras 1', 'AutoKeras 2', 'AutoKeras 3', 'AutoML 1', 'AutoML 2', 'AutoML 3']),
    (['logistic', 'mlp', 'random_forest', 'extra_trees', 'hist_gradient_boosting'],
     ['Logistic', 'MLP', 'Random forest', 'Extra trees', 'Hist. boosting'])]
for j, ((setting, sub), (order, names)) in enumerate(zip(settings, model_groups)):
    d = read(EXP / sub / 'details.csv')
    d = d[(d.target == 'heldout') & (d.metric == 'accuracy')]
    assert d.seed.nunique() == 10 and set(d.model.unique()) == set(order)
    values = d.groupby(['model', 'method']).absolute_error.mean().unstack() * 100
    for y, model in enumerate(order):
        baseline, ssme = values.loc[model, ['labeled_only', 'ssme']]
        axes[j].plot([ssme, baseline], [y, y], color='#BAC4CF', lw=2, zorder=2)
        axes[j].scatter(baseline, y, s=24, color=GRAY, marker='o', zorder=3,
                        label='Labels only' if y == 0 else None)
        axes[j].scatter(ssme, y, s=25, color=BLUE, marker='s', zorder=3,
                        label='SSME-Lite' if y == 0 else None)
        model_rows.append(dict(experiment=setting, model=model, baseline_mae_pp=baseline,
                               ssme_mae_pp=ssme, relative_reduction=1-ssme/baseline, splits=10))
    axes[j].set(yticks=range(len(order)), yticklabels=names, ylim=(len(order)-.5, -.5),
                xlim=(0, 9), xticks=[0, 3, 6, 9], xlabel='Accuracy MAE (pp)')
    axes[j].set_title(['(a) Pretrained checkpoints', '(b) Transfer learning'][j], fontsize=9.5, pad=8)
    axes[j].tick_params(axis='y', length=0, labelsize=8)
    axes[j].tick_params(axis='x', labelsize=8)
    axes[j].spines['left'].set_visible(False)
    axes[j].spines['bottom'].set_color('#CBD5E1')
    axes[j].grid(axis='x', color='#E2E8F0', lw=.65)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.57, 1.05),
           ncol=2, frameon=False, fontsize=8.5, handletextpad=.3, columnspacing=1.5)
pd.DataFrame(model_rows).to_csv(HERE/'model_accuracy_statistics.csv', index=False)
save(fig, 'diagnostics')

# Preserve the selection and sensitivity evidence supporting the text.
selection_rows = []
for setting, _ in settings:
    for metric in ['accuracy', 'auc']:
        d = frames[setting]
        for method, val in d[d.metric == metric].groupby('method').top1_correct.mean().items():
            selection_rows.append(dict(experiment=setting, metric=metric, method=method, top1=val))
pd.DataFrame(selection_rows).to_csv(HERE/'selection_statistics.csv', index=False)
ablation = pd.concat([read(EXP/'transfer/results/nl20_nu400_m5'/f'contribution_seed{s}.csv').assign(seed=s)
                     for s in range(10)], ignore_index=True)
assert len(ablation) == 50 and ablation.epochs.eq(100).all()
sensitivity = []
for model in ['mlp', 'logistic', 'hist_gradient_boosting', 'random_forest', 'extra_trees']:
    mu, half = mean_ci(ablation[ablation.removed_model == model].posterior_total_variation)
    sensitivity.append(dict(model=model, mean_tv=mu, ci_low=mu-half, ci_high=mu+half, seeds=10))
pd.DataFrame(sensitivity).to_csv(HERE/'sensitivity_statistics.csv', index=False)

# Figure 3: broader method comparison and annotation-budget study.
civil_path = EXP.parent / 'reproduction/reference/civilcomments_details.csv'
if not civil_path.exists():
    civil_path = ROOT / 'results/civilcomments_package_half_split/details.csv'
civil = read(civil_path)
civil = civil[civil.model.str.startswith('alg_')]
assert civil.seed.nunique() == 50 and civil.model.nunique() == 7
means = civil.groupby(['method', 'metric']).absolute_error.mean().unstack()
ratios = means / means.loc['labeled_only']
means.to_csv(HERE/'civilcomments_mae.csv')
fig, axes = plt.subplots(1, 2, figsize=(6, 2.0), layout='constrained')
metrics = ['accuracy', 'auc', 'auprc', 'ece']; x = np.arange(4)
for j, (method, label, color) in enumerate([('majority_vote', 'W. vote', '#AEB8C4'), ('pl', 'Pseudo-label', GOLD),
                                           ('dawid_skene', 'Dawid--Skene', PURPLE), ('ssme', 'SSME-Lite', BLUE)]):
    axes[0].bar(x+(j-1.5)*.19, ratios.loc[method, metrics], .18, label=label, color=color)
axes[0].axhline(1, color='#47515B', ls='--', lw=.8)
axes[0].set(xticks=x, xticklabels=['Acc.', 'AUC', 'AP', 'ECE'], ylabel='Relative MAE',
            title='(a) CivilComments quickstart', ylim=(0, 2.35), yticks=[0, 1, 2])
axes[0].legend(ncol=2, loc='upper left', frameon=False, handlelength=1.2,
               handletextpad=.4, columnspacing=.65)
budget = []
for method, label, color in [('labeled_only', 'Labels only', GRAY), ('ssme', 'SSME-Lite', BLUE)]:
    yy, ee = [], []
    for n in [20, 50, 100]:
        d = read(EXP/f'transfer/results/nl{n}_nu400_m5/per_seed.csv')
        values = d[(d.target == 'heldout') & (d.metric == 'accuracy') & (d.method == method)].mae * 100
        mu, half = mean_ci(values); yy.append(mu); ee.append(half)
        budget.append(dict(labels=n, method=method, mae_pp=mu, ci_low=mu-half, ci_high=mu+half))
    axes[1].errorbar([20, 50, 100], yy, yerr=ee, label=label, color=color, marker='o', capsize=3, lw=1.4, ms=4)
axes[1].set(title='(b) PneumoniaMNIST transfer', xlabel='Visible labels (400 unlabeled)',
            ylabel='Accuracy MAE (pp)', xticks=[20, 50, 100], ylim=(0, 11), yticks=[0, 5, 10])
axes[1].legend(frameon=False)
pd.DataFrame(budget).to_csv(HERE/'budget_statistics.csv', index=False)
save(fig, 'evidence')

inputs += [EXP.parent/'ssme_lite/estimation/estimator.py', EXP.parent/'ssme_lite/estimation/contribution.py',
           EXP.parent/'ssme_lite/benchmark/runner.py', EXP/'civilcomments/reproduce_official_split.py',
           EXP/'transfer/transfer_models.py', EXP/'transfer/transfer_ssme.py',
           EXP/'pneumoniamnist/pneumonia_ssme.py', EXP/'pneumoniamnist/pneumonia_models.py',
           EXP/'pneumoniamnist/results/nl20_nu400_m7/config.json',
           EXP/'transfer/results/nl20_nu400_m5/config.json']
manifest = {'note': 'Reanalysis of saved experiments; no classifier training or complete benchmark reruns.',
            'files': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in dict.fromkeys(inputs)],
            'civil_seeds': [int(s) for s in sorted(civil.seed.unique())]}
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Generated 3 vector figures, MAE table, 6 statistical CSVs, and input manifest.')
print(pd.DataFrame(sensitivity).round(5).to_string(index=False))
