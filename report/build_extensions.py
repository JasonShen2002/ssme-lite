"""Create publication figures from archived extension summaries; no model fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT / 'experiments/extensions/results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
TEAL, SLATE, GOLD, INK = '#087F8C', '#7D8998', '#B48235', '#24374B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8.5,'axes.titleweight':'bold','axes.titlecolor':INK,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9AA7B4',
 'axes.linewidth':.6,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'pdf.fonttype':42,'ps.fonttype':42,'legend.fontsize':7.2})
paths = [EXT/'labels/full/summary.csv', EXT/'performance/full/summary.csv',
         EXT/'stress/full/summary.csv']
labels, perf, stress = [pd.read_csv(p) for p in paths]
fig, axs = plt.subplots(2,2,figsize=(6.3,3.25))
fig.subplots_adjust(left=.085,right=.99,bottom=.11,top=.865,wspace=.36,hspace=.85)
methods=[('labeled_only','Labels only',SLATE,'o'),('ssme','SSME-Lite',TEAL,'s'),
         ('mean_probability','Mean probability',GOLD,'^')]
for ax,pipeline,title in zip(axs[0],['checkpoints','transfer'],['(a) Seven checkpoints','(b) Five transfer heads']):
 for method,label,color,marker in methods:
  d=labels.query('pipeline == @pipeline and target == "heldout" and metric == "accuracy" and method == @method').sort_values('budget')
  assert len(d)==3 and d.valid_seeds.eq(30).all()
  ci=t.ppf(.975,d.valid_seeds-1)*d.mae_sd/np.sqrt(d.valid_seeds)*100
  ax.errorbar(d.budget,d.mae*100,yerr=ci,color=color,marker=marker,lw=1.35,ms=3.4,capsize=2,elinewidth=.7,label=label)
 ax.set(title=title,xlabel='Visible labels (fixed 420-row pool)',ylabel='Accuracy MAE (pp)',xticks=[20,50,100],ylim=(0,10),yticks=[0,5,10])
 ax.grid(axis='y',color='#E6EBEF',linewidth=.6)
 ax.set_axisbelow(True)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.005),ncol=3,frameon=False,columnspacing=1.7)
ax=axs[1,0]
d=perf.query('kind == "controlled" and models == 3 and classes == 2 and workers == 1 and batch == 512').sort_values('n')
assert len(d)==4 and d.fit_seconds_count.eq(3).all()
ax.errorbar(d.n,d.fit_seconds_mean,yerr=d.fit_seconds_std,color=TEAL,marker='o',ms=4,lw=1.4,capsize=2)
ax.set(xscale='log',yscale='log',xlabel='Fitted samples (log scale)',ylabel='Fit time (s, log scale)',title='(c) Measured CPU scaling',xticks=d.n,yticks=[3,10,40,250])
ax.set_xticklabels(['500','1,000','2,000','5,000']); ax.set_yticklabels(['3','10','40','250'])
ax.minorticks_off(); ax.grid(color='#E6EBEF',linewidth=.6);ax.set_axisbelow(True)
for _,r in d.iterrows():
 ax.annotate(f'{r.fit_seconds_mean:.1f}s',(r.n,r.fit_seconds_mean),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7,color=INK)
ax.set_ylim(1.6,520)
ax=axs[1,1]
groups=[(3,20),(10,60)]; x=np.arange(2)
series=[('three_diverse','labeled_only','Labels',SLATE),('three_diverse','ssme','3 diverse','#729DAC'),
        ('six_diverse','ssme','6 diverse',TEAL),('duplicated','ssme','3 duplicated',GOLD)]
# Resolve actual recorded identifiers, without altering archived data.
regimes=stress.regime.unique().tolist()
for j,(regime,method,label,color) in enumerate(series):
 if regime=='duplicated':
  regime=next(r for r in regimes if 'dup' in r or 'repeat' in r)
 vals=[]
 for k,budget in groups:
  rows=stress.query('classes == @k and budget == @budget and regime == @regime and method == @method and target == "fixed_pool" and metric == "accuracy"')
  assert len(rows)==1 and int(rows.iloc[0].valid_seeds)==10
  vals.append(rows.iloc[0].mae*100)
 ax.bar(x+(j-1.5)*.19,vals,width=.17,color=color,label=label,zorder=3)
ax.set(title='(d) Model-composition diagnostics',ylabel='Accuracy MAE (pp)',xticks=x,xticklabels=['3 classes / 20 labels','10 classes / 60 labels'],ylim=(0,6),yticks=[0,2,4,6])
ax.grid(axis='y',color='#E6EBEF',linewidth=.6);ax.set_axisbelow(True)
ax.legend(ncol=2,loc='upper left',frameon=False,fontsize=6.7,handlelength=1,handletextpad=.35,columnspacing=.7)
for ext in ['pdf','png']:
 fig.savefig(FIG/f'extensions.{ext}',dpi=240)
plt.close(fig)
manifest={'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
 'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'intervals':'Panels a/b: unadjusted marginal 95% t intervals across 30 seeds; panel c: SD across three process repetitions; panel d: means across ten seeds.',
 'selection':'Panel d displays the two settings with complete class coverage in all ten seeds; full grid remains in experiments/extensions/results/stress/full.'}
(HERE/'extension_figure_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created vector PDF and 240-dpi PNG: figures/extensions; input hashes recorded.')
