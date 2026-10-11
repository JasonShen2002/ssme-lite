"""Isolated, resumable experiments. Run --study all --profile full from a checkout."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.extensions.common import digest, environment, write_json


def conditions(study, profile):
    pilot = profile == 'pilot'
    iterations = 5 if pilot else 100
    jobs = []
    if study == 'labels':
        for pipeline in ('checkpoints', 'transfer'):
            for seed in range(100, 102 if pilot else 130):
                for budget in (20, 50, 100):
                    jobs.append(dict(study=study, pipeline=pipeline, seed=seed, budget=budget,
                                     pool_size=420, iterations=iterations, draws=30))
    elif study == 'stress':
        for k in (3, 10):
            for seed in range(100, 102 if pilot else 110):
                for budget in (20, 60):
                    for regime in ('three_diverse', 'six_diverse', 'three_duplicated'):
                        jobs.append(dict(study=study, classes=k, seed=seed, budget=budget, regime=regime,
                                         pool_size=240, iterations=iterations, draws=30))
    else:
        sizes = [500, 1000] if pilot else [500, 1000, 2000, 5000]
        if profile == 'server':
            sizes += [10000]
        variants = [dict(kind='controlled', n=n, models=3, classes=2, workers=1, batch=512) for n in sizes]
        variants += [dict(kind='controlled', n=1000, models=m, classes=k, workers=w, batch=b)
                     for m, k, w, b in [(3, 2, 2, 512), (3, 2, 1, 128), (6, 2, 1, 512), (3, 5, 1, 512)]]
        variants += [dict(kind='real', n=624, models=7, classes=2, workers=w, batch=512) for w in (1, 2)]
        variants += [dict(kind='prediction', n=n, models=3, classes=2, workers=w, batch=512)
                     for n in ([1000] if pilot else [1000, 5000]) for w in (1, 2)]
        for variant in variants:
            for repetition in range(1 if pilot else 3):
                jobs.append(dict(study=study, **variant, repetition=repetition, iterations=iterations))
    return jobs


def fingerprint(study):
    files = sorted((ROOT / 'ssme_lite').rglob('*.py'))
    files += sorted((ROOT / 'experiments/extensions').glob('*.py'))
    files += [ROOT / 'experiments/extensions/PROTOCOL.md']
    if study in ('labels', 'performance'):
        files += sorted((ROOT / 'reproduction/inputs').glob('*.npz'))
    if study == 'stress':
        files += sorted((ROOT / 'experiments/extensions/inputs').glob('*'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()}
    payload = dict(files=hashes, environment=environment())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def worker(config_path, result_path):
    # Imports/training are outside phase timers; peak RSS includes them.
    import resource
    from threadpoolctl import threadpool_limits
    from experiments.extensions import studies
    job = json.loads(Path(config_path).read_text())
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            result = getattr(studies, job['study'])(job)
    except Exception as exc:
        result = dict(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(job=job, wall_seconds=time.perf_counter()-started,
                  peak_rss_mib=peak / (1024**2 if sys.platform == 'darwin' else 1024))
    write_json(result_path, result)


def run_study(study, args):
    output = args.output / study / args.profile
    output.mkdir(parents=True, exist_ok=True)
    key, provenance = fingerprint(study)
    jobs = conditions(study, args.profile)
    manifest_path = output / 'manifest.json'
    manifest = dict(study=study, profile=args.profile, fingerprint=key, provenance=provenance,
                    expected_conditions=len(jobs), timeout_seconds=args.timeout,
                    execution_jobs=1 if study == 'performance' else args.jobs,
                    protocol_sha256=digest(ROOT / 'experiments/extensions/PROTOCOL.md'), conditions=[])
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['fingerprint'] != key:
            raise ValueError(f'{output}: inputs/code/environment changed; choose a new --output to preserve provenance')
    for name in provenance['files']:
        if name.endswith(('.py', '.md')):
            destination = output/'source_snapshot'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
    write_json(manifest_path, manifest)
    def execute(job):
        ident = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
        config = output / 'jobs' / (ident + '.json')
        result_path = output / 'runs' / (ident + '.json')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['job'] != job:
                raise ValueError('resume configuration mismatch')
        else:
            write_json(config, job)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT/'.mplconfig'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            with (output / 'runs' / (ident + '.log')).open('w') as log:
                try:
                    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config),
                                           '--result', str(result_path)], env=env, stdout=log, stderr=log,
                                          timeout=args.timeout)
                    if proc.returncode != 0 or not result_path.exists():
                        write_json(result_path, dict(job=job, status='failed', error=f'worker exit {proc.returncode}'))
                except subprocess.TimeoutExpired:
                    write_json(result_path, dict(job=job, status='timeout', error=f'exceeded {args.timeout}s'))
            result = json.loads(result_path.read_text())
        return dict(id=ident, status=result['status'], job=job,
                    result=str(result_path.relative_to(output)), sha256=digest(result_path))
    with ThreadPoolExecutor(max_workers=manifest['execution_jobs']) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            item = future.result()
            manifest['conditions'].append(item)
            manifest['conditions'].sort(key=lambda r: r['id'])
            write_json(manifest_path, manifest)
            print(f"{study}/{args.profile} {len(manifest['conditions'])}/{len(jobs)} {item['status']} {item['id']}", flush=True)
    manifest['complete'] = len(manifest['conditions']) == len(jobs)
    manifest['successful_conditions'] = sum(r['status'] == 'ok' for r in manifest['conditions'])
    write_json(manifest_path, manifest)
    return manifest['successful_conditions'] == len(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['all', 'performance', 'labels', 'stress'], default='all')
    parser.add_argument('--profile', choices=['pilot', 'full', 'server'], default='pilot')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/extensions/results')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=600, help='Seconds per condition; timeout remains recorded')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.result)
        return
    if args.jobs < 1 or args.timeout < 1:
        parser.error('jobs and timeout must be positive')
    args.output = args.output.resolve()
    succeeded = []
    for study in (['labels', 'stress', 'performance'] if args.study == 'all' else [args.study]):
        succeeded.append(run_study(study, args))
    if not all(succeeded):
        raise SystemExit('Some workers failed or timed out; inspect the saved manifests and logs.')


if __name__ == '__main__':
    main()
