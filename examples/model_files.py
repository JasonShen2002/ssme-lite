"""Prepare trusted model files and a partial-label JSON workflow; no downloads.

python examples/model_files.py
ssme-lite check-config results/model_files/config.json
ssme-lite results/model_files/config.json --trust-model-files
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ssme_lite.cli import MODEL_TEMPLATE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/model_files'))
    args=parser.parse_args()
    out=args.output
    (out/'models').mkdir(parents=True,exist_ok=True)
    X,y=load_breast_cancer(return_X_y=True)
    train,test=train_test_split(np.arange(len(y)),test_size=.4,stratify=y,random_state=17)
    models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
            'forest':RandomForestClassifier(n_estimators=100,max_depth=6,random_state=17,n_jobs=1)}
    for name,model in models.items():
        model.fit(X[train],y[train])
        joblib.dump(model,out/'models'/f'{name}.joblib')
    partial=np.full(len(test),-1,dtype=int)
    observed=np.random.default_rng(17).permutation(len(test))[:20]
    partial[observed]=y[test[observed]]
    np.savez_compressed(out/'evaluation.npz',X=X[test],y=partial,classes=models['logistic'].classes_,sample_ids=test)
    config=json.loads(json.dumps(MODEL_TEMPLATE))
    config['report']['title']='Trusted model files — breast cancer example'
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps(dict(dataset='sklearn breast_cancer',
       training_indices=train.tolist(),evaluation_indices=test.tolist(),observed_indices=observed.tolist(),
       note='Only visible labels are exported; training is separate from evaluation.'),indent=2)+'\n')
    print(out/'config.json')


if __name__=='__main__':main()
