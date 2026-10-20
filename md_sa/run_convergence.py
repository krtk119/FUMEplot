"""
Compare Sobol-index methods by their convergence under refinement.

For every method in the config's `methods` list and every refinement level, a
full EasyVVUQ campaign is run through the same pipeline as run_campaign.py
(encoder -> runner -> decoder -> analysis -> sobols.yml export). The exported
indices are then compared

  * against a reference (`convergence.reference`):
      analytic   closed-form indices written by a user module (verification)
      finest     the finest level of a named method (no verification case,
                 e.g. MD)
  * against the previous level of the same method ("increment"
    |S_k - S_{k-1}|), a reference-free convergence indicator.

Randomised designs (QMC) are repeated `replicates` times with seeds
`seed`, `seed`+1, ...; errors are reported as median and range.

    python3 run_convergence.py ishigami.yml [--methods PCE-Gauss QMC-Saltelli]
                                            [--max_runs 2000] [--workers 16]
                                            [--out_dir DIR] [--no_plots]

Outputs in <work_dir>/convergence_<timestamp>/ (or --out_dir):

  convergence.csv    one row per (method, level, QoI, quantity): errors,
                     increments, runs, new runs, timings
  summary.yml / .md  per method: runs and spend per level, runs reused,
                     runs to tolerance, empirical rate, computable statistics,
                     declared properties
  sobols/            reference.yml and <method>/<level>.yml (FUMEplot layout)
  plots/             convergence and error plots, Sobol plots per method
"""

import os
import sys
import copy
import time
import argparse
import datetime
import importlib.util
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

from run_campaign import LoadConfig, BuildCampaign, Analyse, ExportSobols, AllQois, SobolsFromDecomposition

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fumeplot'))

from SobolPartition import LoadSobolSet, VarianceMask   # noqa: E402

TINY = 1e-12


# ------------------------------------------------------------------ running

def LevelConfig(cfg, method, level, work_dir, replicate=0):
    c = copy.deepcopy(cfg)
    kwargs = {**(method.get('kwargs') or {}), method['refine']: level}
    if 'seed' in method:                      # randomised designs: one seed per replicate
        kwargs['seed'] = int(method['seed']) + replicate
    c['sampler'] = {'type': method['sampler'], 'kwargs': kwargs,
                    'analysis_kwargs': method.get('analysis_kwargs') or {}}
    c['campaign']['name'] = f"{cfg['campaign']['name']}_{method['name']}_{level}_r{replicate}_"
    c['campaign']['work_dir'] = work_dir
    return c


def ProbeStatistics(results, qoi, params):
    """Which statistics this analysis can deliver."""
    stats = []
    try:
        stats += list(results.supported_stats())
    except Exception:
        pass
    decomposed = SobolsFromDecomposition(results, qoi, params)
    for name, getter in (('sobols_first', lambda: results.sobols_first(qoi, params[0])),
                         ('sobols_second', lambda: results.sobols_second(qoi, params[0])),
                         ('sobols_total', lambda: results.sobols_total(qoi, params[0]))):
        try:
            getter()
            stats.append(name)
        except Exception:
            if decomposed:
                stats.append(name + ' (from full decomposition)')
    if decomposed:
        stats.append('sobols_all_orders')
    for name in ('surrogate',):
        try:
            results.surrogate()
            stats.append(name)
        except Exception:
            pass
    return stats


def RunLevel(cfg, cfg_dir, method, level, out_dir, workers, replicate=0):
    c = LevelConfig(cfg, method, level, os.path.join(out_dir, 'campaigns'), replicate)
    params = list(cfg['uncertain'])

    t0 = time.time()
    campaign = BuildCampaign(c, cfg_dir)
    with ThreadPoolExecutor(workers) as pool:
        campaign.execute(pool=pool).collate()
    t1 = time.time()
    results = Analyse(campaign, c)
    t2 = time.time()

    yml = os.path.join(out_dir, 'sobols', method['name'], f"{level}.yml" if replicate == 0 else f"{level}_r{replicate}.yml")
    Path(yml).parent.mkdir(parents=True, exist_ok=True)
    ExportSobols(results, campaign, c, yml)

    runs = list(campaign.list_runs())
    points = {tuple(round(float(r['params'][p]), 10) for p in params) for _, r in runs}
    run_times = []
    for _, r in runs:
        try:
            with open(os.path.join(r['run_dir'], cfg['application']['output_file'])) as f:
                run_times.append(float(yaml.safe_load(f).get('meta', {}).get('wall_time', np.nan)))
        except Exception:
            run_times.append(np.nan)

    return {
        'method': method['name'], 'level': level, 'replicate': replicate, 'yml': yml, 'points': points,
        'n_runs': len(runs), 't_execute': t1 - t0, 't_analyse': t2 - t1,
        'cpu_model': float(np.nansum(run_times)),
        'statistics': ProbeStatistics(results, AllQois(cfg)[0], params),
    }


def WriteReference(cfg, cfg_dir, level_records, out_dir):
    ref = cfg.get('convergence', {}).get('reference', {'type': 'finest'})
    path = os.path.join(out_dir, 'sobols', 'reference.yml')
    if ref['type'] == 'analytic':
        spec = importlib.util.spec_from_file_location('reference_module', os.path.join(cfg_dir, ref['module']))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with open(level_records[0]['yml']) as f:
            info = yaml.safe_load(f)['campaign_info']
        getattr(module, ref['writer'])(path, info, **(ref.get('kwargs') or {}))
        return path, 'analytic'
    name = ref.get('method') or level_records[-1]['method']
    finest = [r for r in level_records if r['method'] == name and r['replicate'] == 0][-1]
    with open(finest['yml']) as f, open(path, 'w') as g:
        g.write(f.read())
    return path, f"finest: {name} level {finest['level']}"


# ------------------------------------------------------------------ errors

def _quantities(qoi, params):
    """(quantity, label, array) for every statistic of a SobolQoi."""
    out = []
    for p in params:
        out.append(('S1', p, qoi.first.get(p)))
        out.append(('ST', p, qoi.total.get(p)))
    for pq in combinations(params, 2):
        out.append(('S2', ' x '.join(pq), qoi.second.get(pq)))
    out.append(('mean', '', qoi.mean))
    out.append(('std', '', qoi.std))
    return [(k, l, v) for k, l, v in out if v is not None]


def _norms(est, ref, valid):
    e = (est - ref)[valid]
    abs_rms = float(np.sqrt(np.mean(e**2))) if e.size else np.nan
    ref_rms = float(np.sqrt(np.mean(ref[valid]**2))) if e.size else np.nan
    return abs_rms, (float(np.max(np.abs(e))) if e.size else np.nan), \
        (abs_rms / ref_rms if ref_rms > TINY else np.nan)


def ErrorRows(level_records, ref_set, params):
    rows = []
    previous = {}
    for rec in level_records:
        est = LoadSobolSet(rec['yml'], params)
        chain = (rec['method'], rec['replicate'])
        prev = previous.get(chain)
        for name, qref in ref_set.qois.items():
            if name not in est.qois:
                continue
            valid = ~VarianceMask(qref)
            qest = est.qois[name]
            ref_q = {(k, l): v for k, l, v in _quantities(qref, params)}
            prev_q = {(k, l): v for k, l, v in _quantities(prev.qois[name], params)} if prev else {}
            for k, l, v in _quantities(qest, params):
                if (k, l) not in ref_q:
                    continue
                err_abs, err_max, err_rel = _norms(v, ref_q[(k, l)], valid)
                inc = _norms(v, prev_q[(k, l)], valid)[1] if (k, l) in prev_q else np.nan
                rows.append({
                    'method': rec['method'], 'level': rec['level'], 'replicate': rec['replicate'],
                    'n_runs': rec['n_runs'],
                    'new_runs': rec['new_runs'], 't_execute': rec['t_execute'], 't_analyse': rec['t_analyse'],
                    'qoi': name, 'ndim': qref.ndim, 'quantity': k, 'index': l,
                    'err_abs': err_abs, 'err_max': err_max, 'err_rel': err_rel, 'increment': inc,
                    'estimate': float(v) if qref.ndim == 0 else np.nan,
                    'reference': float(ref_q[(k, l)]) if qref.ndim == 0 else np.nan,
                })
        previous[chain] = est
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ summary

def _sobolErrors(df):
    """
    Worst Sobol-index error (max abs over QoIs, indices and grid) per (method, level):
    median over replicates, with the replicate range.
    """
    s = df[df.quantity.isin(['S1', 'S2', 'ST'])]
    per_rep = s.groupby(['method', 'level', 'n_runs', 'replicate'], sort=False).agg(
        err=('err_max', 'max'), increment=('increment', 'max')).reset_index()
    return per_rep.groupby(['method', 'level', 'n_runs'], sort=False).agg(
        err=('err', 'median'), err_lo=('err', 'min'), err_hi=('err', 'max'),
        increment=('increment', 'median')).reset_index()


def Summarise(df, level_records, cfg, reference_kind):
    conv = cfg.get('convergence', {})
    tols = conv.get('tolerances', [0.1, 0.01])
    run_cost = conv.get('reference_run_cost_s')
    worst = _sobolErrors(df)
    summary = {'reference': reference_kind, 'created': datetime.datetime.now().isoformat(timespec='seconds'),
               'tolerance_metric': 'max abs error over all Sobol indices, QoIs and grid points',
               'methods': {}}
    for m in cfg['methods']:
        recs = [r for r in level_records if r['method'] == m['name'] and r['replicate'] == 0]
        if not recs:
            continue
        w = worst[worst.method == m['name']]
        good = w[w.err > TINY]
        tail = good.iloc[len(good) // 2:] if len(good) >= 3 else good
        rate = float(np.polyfit(np.log(tail.n_runs), np.log(tail.err), 1)[0]) if len(tail) >= 2 else None
        cumulative_new = int(sum(r['new_runs'] for r in recs))
        summary['methods'][m['name']] = {
            'sampler': m['sampler'], 'refine': m['refine'],
            'replicates': int(m.get('replicates', 1)),
            'levels': [r['level'] for r in recs],
            'n_runs': [r['n_runs'] for r in recs],
            'new_runs': [r['new_runs'] for r in recs],
            'runs_reused_total': int(sum(r['n_runs'] - r['new_runs'] for r in recs)),
            'runs_finest': recs[-1]['n_runs'],
            'runs_cumulative_refinement': cumulative_new,
            't_execute_s': [round(r['t_execute'], 2) for r in recs],
            't_analyse_s': [round(r['t_analyse'], 2) for r in recs],
            'model_cpu_s_finest': round(recs[-1]['cpu_model'], 3),
            'projected_md_cpu_h_finest': round(recs[-1]['n_runs'] * run_cost / 3600, 2) if run_cost else None,
            'max_sobol_error': [float(e) for e in w.err],
            'max_sobol_error_range': [[float(a), float(b)] for a, b in zip(w.err_lo, w.err_hi)],
            'max_sobol_increment': [float(e) for e in w.increment],
            'runs_to_tolerance': {str(t): (int(w[w.err <= t].n_runs.min()) if (w.err <= t).any() else None)
                                  for t in tols},
            'empirical_rate': rate,
            'statistics': recs[-1]['statistics'],
            'properties': m.get('properties', {}),
        }
    return summary


def WriteSummaryMarkdown(summary, path):
    tols = list(next(iter(summary['methods'].values()))['runs_to_tolerance'])
    head = ['Method', 'Runs (finest)', 'Runs reused', 'Max Sobol err (finest)', 'Replicates', 'Rate'] + \
           [f"Runs to {t}" for t in tols] + ['Analysis s (finest)', 'Proj. MD CPU-h', 'Second order', 'Surrogate']
    lines = [f"Reference: {summary['reference']}. Error metric: {summary['tolerance_metric']}; "
             f"median [min, max] over replicates for randomised designs.", '',
             '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for name, m in summary['methods'].items():
        rate = f"{m['empirical_rate']:.2f}" if m['empirical_rate'] is not None else '-'
        lo, hi = m['max_sobol_error_range'][-1]
        err = f"{m['max_sobol_error'][-1]:.2e}" + (f" [{lo:.1e}, {hi:.1e}]" if m['replicates'] > 1 else '')
        row = [name, m['runs_finest'], m['runs_reused_total'], err, m['replicates'], rate] + \
              [m['runs_to_tolerance'][t] or '-' for t in tols] + \
              [m['t_analyse_s'][-1], m['projected_md_cpu_h_finest'] or '-',
               'yes' if any(x.startswith('sobols_second') for x in m['statistics']) else 'no',
               'yes' if 'surrogate' in m['statistics'] else 'no']
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    lines += ['', '| Method | Design | Extendability | Statistics |', '|---|---|---|---|']
    for name, m in summary['methods'].items():
        p = m['properties']
        lines.append(f"| {name} | {p.get('design', '')} | {p.get('extendable', '')} | {', '.join(m['statistics'])} |")
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ driver

def RunConvergence(config, methods=None, max_runs=None, workers=None, out_dir=None, plots=True):
    cfg_dir = os.path.dirname(os.path.abspath(config))
    cfg = LoadConfig(config)
    if methods:
        cfg['methods'] = [m for m in cfg['methods'] if m['name'] in methods]
    workers = workers or cfg.get('convergence', {}).get('n_workers', 8)
    out_dir = os.path.abspath(out_dir or os.path.join(
        cfg_dir, cfg['campaign']['work_dir'], f"convergence_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for m in cfg['methods']:
        for replicate in range(int(m.get('replicates', 1))):
            seen = set()
            for level in m['levels']:
                rec = RunLevel(cfg, cfg_dir, m, level, out_dir, workers, replicate)
                rec['new_runs'] = len(rec['points'] - seen)
                seen |= rec['points']
                records.append(rec)
                print(f"[INFO] {m['name']:14s} r{replicate} {m['refine']}={level:<5} runs={rec['n_runs']:<6} "
                      f"new={rec['new_runs']:<6} exec={rec['t_execute']:.1f}s analyse={rec['t_analyse']:.1f}s")
                if max_runs and rec['n_runs'] >= max_runs:
                    break

    ref_path, ref_kind = WriteReference(cfg, cfg_dir, records, out_dir)
    params = list(cfg['uncertain'])
    df = ErrorRows(records, LoadSobolSet(ref_path, params), params)
    df.to_csv(os.path.join(out_dir, 'convergence.csv'), index=False)

    summary = Summarise(df, records, cfg, ref_kind)
    with open(os.path.join(out_dir, 'summary.yml'), 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False, width=120)
    WriteSummaryMarkdown(summary, os.path.join(out_dir, 'summary.md'))
    print(f"[INFO] convergence results in {out_dir}")

    if plots:
        from PlotSobolConvergence import plotSobolConvergence
        plotSobolConvergence(out_dir)
    return out_dir, df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convergence comparison of Sobol-index methods")
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "ishigami.yml"))
    ap.add_argument("--methods", nargs="*", default=None, help="subset of method names")
    ap.add_argument("--max_runs", type=int, default=None, help="stop refining a method beyond this many runs")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    RunConvergence(args.config, args.methods, args.max_runs, args.workers, args.out_dir, not args.no_plots)
