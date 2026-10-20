"""
Second-order Sobol study of the MD model, with its verification evidence.

    python3 run_second_order.py [second_order.yml] [--workers N] [--set n_steps=600 ...]
                                [--levels 1 2] [--out_dir DIR] [--analyse_only DIR] [--no_plots]

Runs a refinement ladder of PCE campaigns (by default p = 2, 3, 4: 81, 256 and
625 MD runs) and repeats the finest design with other engine seeds. Each
campaign goes through the same pipeline as run_campaign.py and is exported as a
FUMEplot sobols.yml (first, second, total order). On top of the convergence
outputs of run_convergence.py (reference = finest level), it writes

  verification.yml / .md   per QoI and per input pair:
      consistency    0 <= S_i, S_ij;  S_i + sum_j S_ij <= S_Ti;  sum S_i + sum S_ij <= 1
      seed           |S(seed a) - S(seed b)| at the finest level, and the noise
                     ratio eta = Var_seed[Y] / Var[Y] (the share of output variance
                     that is trajectory noise rather than input dependence)
      refinement     |S(p) - S(p-1)|
      surrogate_mc   |S_PCE - S_MC|, Saltelli/Jansen estimator on the PCE surrogate
      holdout        Q^2 of the finest surrogate on the (distinct) nodes of the
                     coarser levels, against its ceiling 1 - eta
      null_controls  indices of inputs that cannot affect a QoI's expectation
      thermostat     |<T_kin> - T| / T over all runs
      resolved       S_ij > resolve_factor * max(seed, refinement)
  plots/second_order_parity.png   S from the finest design vs the other estimates

Outputs go to <work_dir>/second_order_<timestamp>/ (or --out_dir).
"""

import os
import sys
import copy
import glob
import json
import time
import argparse
import datetime
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yaml

from run_campaign import LoadConfig, BuildCampaign, ReloadCampaign, Analyse, ExportSobols, AllQois
from run_convergence import (LevelConfig, ProbeStatistics, WriteReference, ErrorRows, Summarise,
                             WriteSummaryMarkdown)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fumeplot'))

from SobolPartition import LoadSobolSet, VarianceMask, PARAM_COLOURS   # noqa: E402


# ------------------------------------------------------------------ config

def _deepUpdate(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deepUpdate(base[k], v)
        else:
            base[k] = v
    return base


def LoadStudy(path, overrides=()):
    """The study file names a `base` campaign config and overrides parts of it."""
    with open(path) as f:
        study = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(path))
    cfg = LoadConfig(os.path.join(cfg_dir, study.pop('base', 'campaign.yml')), overrides)
    return _deepUpdate(cfg, study), cfg_dir


# ------------------------------------------------------------------ running

def ReadRuns(campaign, cfg):
    """Inputs and decoded outputs of every run: X (n, d), {qoi: (n, m)}, temp_kin, wall_time."""
    params = list(cfg['uncertain'])
    X, Y, temp, wall, T_set = [], {q: [] for q in AllQois(cfg)}, [], [], []
    for _, run in campaign.list_runs():
        with open(os.path.join(run['run_dir'], cfg['application']['output_file'])) as f:
            out = json.load(f)
        X.append([float(run['params'][p]) for p in params])
        for q in Y:
            Y[q].append(np.ravel(out[q]))
        temp.append(out.get('temp_kin', np.nan))
        wall.append(out.get('meta', {}).get('wall_time', np.nan))
        T_set.append(float(run['params'].get('temperature', np.nan)))
    return {'X': np.array(X), 'Y': {q: np.array(v) for q, v in Y.items()},
            'temp_kin': np.array(temp, dtype=float), 'temperature': np.array(T_set),
            'wall_time': np.array(wall, dtype=float)}


def RunStudyLevel(cfg, cfg_dir, method, level, out_dir, workers, seed=None, replicate=0, reuse=False):
    c = LevelConfig(cfg, method, level, os.path.join(out_dir, 'campaigns'), replicate)
    if seed is not None:
        c['parameters']['seed']['default'] = int(seed)
    params = list(cfg['uncertain'])

    t0 = time.time()
    existing = sorted(glob.glob(os.path.join(c['campaign']['work_dir'], c['campaign']['name'] + '*')))
    if reuse and existing:
        campaign = ReloadCampaign(c, existing[-1])
    else:
        campaign = BuildCampaign(c, cfg_dir)
        with ThreadPoolExecutor(workers) as pool:
            campaign.execute(pool=pool).collate()
    t1 = time.time()
    results = Analyse(campaign, c)
    t2 = time.time()

    name = f"{level}.yml" if replicate == 0 else f"{level}_r{replicate}.yml"
    yml = os.path.join(out_dir, 'sobols', method['name'], name)
    Path(yml).parent.mkdir(parents=True, exist_ok=True)
    ExportSobols(results, campaign, c, yml)

    runs = ReadRuns(campaign, c)
    return {
        'method': method['name'], 'level': level, 'replicate': replicate,
        'seed': int(c['parameters']['seed']['default']), 'yml': yml,
        'points': {tuple(np.round(x, 10)) for x in runs['X']},
        'n_runs': len(runs['X']), 't_execute': t1 - t0, 't_analyse': t2 - t1,
        'cpu_model': float(np.nansum(runs['wall_time'])),
        'statistics': ProbeStatistics(results, AllQois(c)[0], params),
        'results': results, 'runs': runs,
    }


# ------------------------------------------------------------------ verification

def _pairs(params):
    return list(combinations(params, 2))


def RelevantMask(qoi, rel_std):
    """True where the QoI std is at least `rel_std` of its maximum (ratios to Var[Y] are meaningful)."""
    std = np.nan_to_num(np.asarray(qoi.std, dtype=float)) if qoi.std is not None else np.ones(qoi.shape)
    return std >= rel_std * max(np.max(std), np.finfo(float).tiny)


def _maxValid(a, valid):
    a = np.asarray(a, dtype=float)[valid]
    return float(np.max(a)) if a.size else np.nan


def _maxAbsDiff(a, b, valid):
    return _maxValid(np.abs(np.asarray(a) - np.asarray(b)), valid)


def Consistency(sset, tol):
    """Algebraic identities every variance decomposition must satisfy."""
    out = {}
    params = sset.params
    for name, q in sset.qois.items():
        valid = ~VarianceMask(q)
        s1 = np.array([q.first[p] for p in params])
        s2 = np.array([q.second[pq] for pq in _pairs(params) if pq in q.second]) if q.second else np.zeros((0,) + q.shape)
        st = np.array([q.total[p] for p in params])
        pair_sum = np.array([sum((q.second.get(tuple(sorted((p, o), key=params.index)), 0)
                                  for o in params if o != p), np.zeros(q.shape)) for p in params])
        remainder = 1.0 - s1.sum(axis=0) - s2.sum(axis=0)
        checks = {
            'min_S1': _maxValid(-s1.min(axis=0), valid) * -1,
            'min_S2': (_maxValid(-s2.min(axis=0), valid) * -1) if len(s2) else np.nan,
            'max_S1_plus_pairs_minus_ST': _maxValid((s1 + pair_sum - st).max(axis=0), valid),
            'max_sum_minus_1': _maxValid(-remainder, valid),
            'remainder_R_max': _maxValid(remainder, valid),
        }
        checks['pass'] = bool(checks['min_S1'] >= -tol
                              and (np.isnan(checks['min_S2']) or checks['min_S2'] >= -tol)
                              and checks['max_S1_plus_pairs_minus_ST'] <= tol
                              and checks['max_sum_minus_1'] <= tol)
        out[name] = checks
    return out


def IndexDifferences(a, b, params):
    """{qoi: {quantity: {label: (max |S_a - S_b|, elementwise |S_a - S_b|)}}} over the valid grid of `a`."""
    out = {}
    for name, qa in a.qois.items():
        if name not in b.qois:
            continue
        qb = b.qois[name]
        valid = ~VarianceMask(qa)
        d = {'S1': {}, 'ST': {}, 'S2': {}}
        for p in params:
            d['S1'][p] = np.abs(qa.first[p] - qb.first[p])
            d['ST'][p] = np.abs(qa.total[p] - qb.total[p])
        # an estimate without interactions (e.g. PCE p = 1) says nothing about S_ij
        has_pairs = any(np.any(v != 0) for v in qb.second.values())
        for pq in _pairs(params):
            if has_pairs and pq in qa.second and pq in qb.second:
                d['S2'][' x '.join(pq)] = np.abs(qa.second[pq] - qb.second[pq])
        out[name] = {k: {l: (_maxValid(v, valid), v) for l, v in vs.items()} for k, vs in d.items()}
    return out


def SeedNoise(runs_a, runs_b, sset, rel_std=0.05):
    """eta = Var_seed[Y] / Var[Y]: share of the output variance due to trajectory noise."""
    key_b = {tuple(np.round(x, 10)): i for i, x in enumerate(runs_b['X'])}
    pairs = [(i, key_b[tuple(np.round(x, 10))]) for i, x in enumerate(runs_a['X'])
             if tuple(np.round(x, 10)) in key_b]
    ia, ib = map(np.array, zip(*pairs))
    out = {}
    for name, q in sset.qois.items():
        ya, yb = runs_a['Y'][name][ia], runs_b['Y'][name][ib]
        var_seed = 0.5 * np.mean((ya - yb)**2, axis=0).reshape(q.shape)
        var_y = np.asarray(q.std, dtype=float)**2
        valid = ~VarianceMask(q) & RelevantMask(q, rel_std)
        eta = np.where(valid, var_seed / np.where(valid, var_y, 1.0), np.nan)
        out[name] = {'eta_median': float(np.nanmedian(eta)) if valid.any() else np.nan,
                     'eta_max': float(np.nanmax(eta)) if valid.any() else np.nan,
                     'n_pairs': len(pairs), 'eta': eta}
    return out


def _evalFit(results, qoi, X):
    """Evaluate the chaospy PCE fit of `qoi` at X (n, d) -> (n, m)."""
    y = np.asarray(results.raw_data['fit'][qoi](*X.T), dtype=float)
    return y.reshape(-1, X.shape[0]).T


def Holdout(results, held_runs, sset, rel_std=0.05):
    """Q^2 of the surrogate on runs it was not fitted to (per grid point; median, min, pooled)."""
    out = {}
    X = held_runs['X']
    for name, q in sset.qois.items():
        y = held_runs['Y'][name]
        yhat = _evalFit(results, name, X)
        ss_res = np.sum((y - yhat)**2, axis=0)
        ss_tot = np.sum((y - y.mean(axis=0))**2, axis=0)
        valid = (~VarianceMask(q) & RelevantMask(q, rel_std)).ravel() & \
            (ss_tot > 1e-12 * max(ss_tot.max(), 1e-300))
        q2 = np.where(valid, 1.0 - ss_res / np.where(valid, ss_tot, 1.0), np.nan)
        out[name] = {'n_points': int(X.shape[0]),
                     'q2_median': float(np.nanmedian(q2)) if valid.any() else np.nan,
                     'q2_min': float(np.nanmin(q2)) if valid.any() else np.nan,
                     'q2_pooled': float(1.0 - ss_res[valid].sum() / ss_tot[valid].sum()) if valid.any() else np.nan}
    return out


def SurrogateMC(results, cfg, sset, n_base, seed=0):
    """Saltelli/Jansen estimates (SALib) of S_i, S_ij, S_Ti on the PCE surrogate, scalar QoIs."""
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    params = list(cfg['uncertain'])
    if any(cfg['uncertain'][p]['dist'] != 'Uniform' for p in params):
        print("[INFO] surrogate MC check supports Uniform inputs only; skipped")
        return {}
    problem = {'num_vars': len(params), 'names': params,
               'bounds': [cfg['uncertain'][p]['args'] for p in params]}
    X = sobol_sample.sample(problem, n_base, calc_second_order=True, seed=seed)
    order = [params.index(p) for p in results.inputs]       # fit argument order
    out = {}
    for q in sset.byDim(0):
        y = _evalFit(results, q.name, X[:, order]).ravel()
        r = sobol_analyze.analyze(problem, y, calc_second_order=True, seed=seed, print_to_console=False)
        out[q.name] = {
            'S1': {p: (float(r['S1'][i]), float(r['S1_conf'][i])) for i, p in enumerate(params)},
            'ST': {p: (float(r['ST'][i]), float(r['ST_conf'][i])) for i, p in enumerate(params)},
            'S2': {f"{a} x {b}": (float(r['S2'][i, j]), float(r['S2_conf'][i, j]))
                   for (i, a), (j, b) in combinations(enumerate(params), 2)},
        }
    return out


def NullControls(sset, controls):
    """Largest index of an input on QoIs whose expectation it cannot change."""
    out = {}
    for p, qois in (controls or {}).items():
        if p not in sset.params:
            continue
        for name in qois:
            if name not in sset.qois:
                continue
            q = sset.qois[name]
            valid = ~VarianceMask(q)
            s2 = [v for pq, v in q.second.items() if p in pq]
            out[f"{p} on {name}"] = {
                'max_S1': _maxValid(np.abs(q.first[p]), valid),
                'max_ST': _maxValid(q.total[p], valid),
                'max_S2': max((_maxValid(np.abs(v), valid) for v in s2), default=np.nan),
            }
    return out


def Thermostat(records, rel_tol):
    temp = np.concatenate([r['runs']['temp_kin'] for r in records])
    T = np.concatenate([r['runs']['temperature'] for r in records])
    dev = (temp - T) / T
    return {'n_runs': int(np.isfinite(dev).sum()), 'rel_tol': rel_tol,
            'max_rel_dev': float(np.nanmax(np.abs(dev))), 'mean_rel_dev': float(np.nanmean(dev)),
            'n_exceeding': int(np.sum(np.abs(dev) > rel_tol)), 'pass': bool(np.nanmax(np.abs(dev)) <= rel_tol)}


def Resolution(fine, seed_diff, refine_diff, factor):
    """
    Per QoI and pair: largest S_ij, its uncertainty u = max(seed, refinement)
    differences, and the fraction of the valid grid where S_ij > factor * u.
    """
    out = {}
    for name, q in fine.qois.items():
        valid = ~VarianceMask(q)
        rows = {}
        for pq, s2 in q.second.items():
            label = ' x '.join(pq)
            u = np.zeros(q.shape)
            for diffs in (seed_diff, refine_diff):
                if diffs and name in diffs and label in diffs[name]['S2']:
                    u = np.maximum(u, diffs[name]['S2'][label][1])
            resolved = (s2 > factor * u) & valid
            rows[label] = {'max_S2': _maxValid(s2, valid),
                           'max_uncertainty': _maxValid(u, valid),
                           'resolved_fraction': float(resolved.sum() / max(valid.sum(), 1))}
        out[name] = rows
    return out


def _clean(obj):
    """Drop elementwise arrays and convert numpy scalars for YAML."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], np.ndarray):
        return _clean(obj[0])
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    return obj


def _f(x, fmt='.3f'):
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def WriteVerificationMarkdown(v, fine, path):
    params = fine.params
    L = [f"# Second-order Sobol verification: {v['design']}", '',
         f"Reference estimate: {v['reference']}. Uncertainty of S_ij: u = max(|seed difference|, "
         f"|refinement increment|); resolved where S_ij > {v['resolve_factor']} u.", '',
         f"Thermostat: max |<T_kin> - T|/T = {_f(v['thermostat']['max_rel_dev'], '.4f')}, mean signed "
         f"{_f(v['thermostat']['mean_rel_dev'], '+.4f')}; {v['thermostat']['n_exceeding']} of "
         f"{v['thermostat']['n_runs']} runs above {v['thermostat']['rel_tol']}.", '',
         "eta and Q2 are taken over grid points with std >= relevant_std * max(std). "
         "Q2 cannot exceed about 1 - eta: the remaining misfit is trajectory noise, not surrogate error.", '',
         '## Per QoI', '',
         '| QoI | consistency | R max | eta median | eta max | holdout Q2 (median) | 1 - eta max | '
         'max seed dS | max refine dS | max MC dS |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for name in fine.qois:
        c = v['consistency'].get(name, {})
        s = v['seed_noise'].get(name, {})
        h = v['holdout'].get(name, {})

        def worst(d):
            if not d or name not in d:
                return np.nan
            return max((x[0] for qs in d[name].values() for x in qs.values()), default=np.nan)

        mc = v['surrogate_mc'].get(name)
        mc_err = np.nan
        if mc:
            q = fine.qois[name]
            ests = [(mc['S1'][p][0], float(q.first[p])) for p in params] + \
                   [(mc['ST'][p][0], float(q.total[p])) for p in params] + \
                   [(mc['S2'][' x '.join(pq)][0], float(q.second[pq])) for pq in q.second]
            mc_err = max(abs(a - b) for a, b in ests)
        L.append(f"| {name} | {'pass' if c.get('pass') else 'FAIL'} | {_f(c.get('remainder_R_max'))} | "
                 f"{_f(s.get('eta_median'), '.2e')} | {_f(s.get('eta_max'), '.2e')} | "
                 f"{_f(h.get('q2_median'))} | {_f(1 - s['eta_max'] if s else np.nan)} | "
                 f"{_f(worst(v['_seed_diff']))} | {_f(worst(v['_refine_diff']))} | {_f(mc_err, '.1e')} |")

    L += ['', '## Pairwise indices', '',
          '| QoI | pair | max S_ij | max u | resolved fraction |', '|---|---|---|---|---|']
    for name, rows in v['resolution'].items():
        for label, r in rows.items():
            L.append(f"| {name} | {label} | {_f(r['max_S2'])} | {_f(r['max_uncertainty'])} | "
                     f"{_f(r['resolved_fraction'], '.2f')} |")

    if v['null_controls']:
        L += ['', '## Negative controls (should be at the noise floor)', '',
              '| control | max S_i | max S_Ti | max S_ij |', '|---|---|---|---|']
        for k, r in v['null_controls'].items():
            L.append(f"| {k} | {_f(r['max_S1'])} | {_f(r['max_ST'])} | {_f(r['max_S2'])} |")
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def PlotParity(fine, others, mc, folder):
    """Scalar QoIs: S from the finest design (x) against every other estimate (y)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    params = fine.params
    scalars = fine.byDim(0)
    if not scalars:
        return
    panels = [(label, s) for label, s in others.items()] + ([('surrogate MC', None)] if mc else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    kinds = [('S1', 'o', 'First order $S_i$'), ('S2', 's', 'Second order $S_{ij}$'), ('ST', '^', 'Total $S_{Ti}$')]
    for ax, (label, other) in zip(axes[0], panels):
        for k, (kind, marker, name) in enumerate(kinds):
            xs, ys = [], []
            for q in scalars:
                if kind == 'S1':
                    items = [(q.first[p], (other.qois[q.name].first[p] if other else mc[q.name]['S1'][p][0]))
                             for p in params]
                elif kind == 'ST':
                    items = [(q.total[p], (other.qois[q.name].total[p] if other else mc[q.name]['ST'][p][0]))
                             for p in params]
                else:
                    items = [(v, (other.qois[q.name].second[pq] if other else mc[q.name]['S2'][' x '.join(pq)][0]))
                             for pq, v in q.second.items()]
                xs += [float(a) for a, _ in items]
                ys += [float(b) for _, b in items]
            ax.scatter(xs, ys, s=36, marker=marker, color=PARAM_COLOURS[k], edgecolor='white',
                       linewidth=0.8, label=name, zorder=3)
        ax.plot([0, 1], [0, 1], color='#52514e', linewidth=1, zorder=2)
        ax.set_xscale('symlog', linthresh=1e-2, linscale=0.5)
        ax.set_yscale('symlog', linthresh=1e-2, linscale=0.5)
        ax.set(xlim=(-2e-3, 1), ylim=(-2e-3, 1), xlabel='finest design', ylabel=label, title=f"vs {label}")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.01, 0.1, 1]))
            a.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.33)
    axes[0][0].legend(loc='upper left', fontsize='small')
    fig.suptitle('Sobol indices of scalar QoIs: agreement between estimates (symlog, linear below 0.01)',
                 fontsize='medium')
    fig.tight_layout()
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(folder, 'second_order_parity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ driver

def RunSecondOrder(config, overrides=(), levels=None, workers=None, out_dir=None, reuse=False, plots=True):
    cfg, cfg_dir = LoadStudy(config, overrides)
    method = cfg['methods'][0]
    if levels:
        method['levels'] = levels
    so = cfg.get('second_order', {})
    workers = workers or cfg.get('convergence', {}).get('n_workers', 8)
    out_dir = os.path.abspath(out_dir or os.path.join(
        cfg_dir, cfg['campaign']['work_dir'], f"second_order_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, 'study_config.yml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=120)

    base_seed = int(cfg['parameters']['seed']['default'])
    records, seen = [], set()
    for level in method['levels']:
        rec = RunStudyLevel(cfg, cfg_dir, method, level, out_dir, workers, seed=base_seed, reuse=reuse)
        rec['new_runs'] = len(rec['points'] - seen)
        seen |= rec['points']
        records.append(rec)
        print(f"[INFO] p={level} seed={base_seed}: {rec['n_runs']} runs, exec {rec['t_execute']:.0f}s, "
              f"analyse {rec['t_analyse']:.0f}s")
    replicas = []
    for k, seed in enumerate(so.get('seed_replicates', []), start=1):
        rec = RunStudyLevel(cfg, cfg_dir, method, method['levels'][-1], out_dir, workers,
                            seed=seed, replicate=k, reuse=reuse)
        rec['new_runs'] = rec['n_runs']
        replicas.append(rec)
        print(f"[INFO] p={rec['level']} seed={seed}: {rec['n_runs']} runs, exec {rec['t_execute']:.0f}s")

    # convergence outputs (reference = finest level, seed 0 chain only)
    params = list(cfg['uncertain'])
    ref_path, ref_kind = WriteReference(cfg, cfg_dir, records, out_dir)
    df = ErrorRows(records, LoadSobolSet(ref_path, params), params)
    df.to_csv(os.path.join(out_dir, 'convergence.csv'), index=False)
    summary = Summarise(df, records, cfg, ref_kind)
    with open(os.path.join(out_dir, 'summary.yml'), 'w') as f:
        yaml.safe_dump(summary, f, sort_keys=False, width=120)
    WriteSummaryMarkdown(summary, os.path.join(out_dir, 'summary.md'))

    # verification
    finest = records[-1]
    fine = LoadSobolSet(finest['yml'], params)
    coarser = LoadSobolSet(records[-2]['yml'], params) if len(records) > 1 else None
    seed_sets = [LoadSobolSet(r['yml'], params) for r in replicas]

    held = [r['runs'] for r in records[:-1]]
    held_runs = None
    if held:
        fine_pts = finest['points']
        X = np.vstack([h['X'] for h in held])
        keep = np.array([tuple(np.round(x, 10)) not in fine_pts for x in X])
        held_runs = {'X': X[keep], 'Y': {q: np.vstack([h['Y'][q] for h in held])[keep] for q in fine.qois}}

    mc_cfg = so.get('surrogate_mc') or {}
    rel_std = float(so.get('relevant_std', 0.05))
    v = {
        'design': f"{method['sampler']} p={finest['level']} ({finest['n_runs']} runs, seed {finest['seed']})"
                  + (f", seed replicates {[r['seed'] for r in replicas]}" if replicas else ''),
        'reference': ref_kind,
        'resolve_factor': float(so.get('resolve_factor', 2.0)),
        'consistency': Consistency(fine, float(so.get('tolerance', 1e-6))),
        'consistency_seed_replicates': [Consistency(s, float(so.get('tolerance', 1e-6))) for s in seed_sets],
        '_seed_diff': IndexDifferences(fine, seed_sets[0], params) if seed_sets else {},
        '_refine_diff': IndexDifferences(fine, coarser, params) if coarser else {},
        'seed_noise': SeedNoise(finest['runs'], replicas[0]['runs'], fine, rel_std) if replicas else {},
        'holdout': Holdout(finest['results'], held_runs, fine, rel_std) if held_runs and len(held_runs['X']) else {},
        'surrogate_mc': SurrogateMC(finest['results'], cfg, fine, int(mc_cfg.get('n_base', 4096)),
                                    int(mc_cfg.get('seed', 0))) if mc_cfg else {},
        'null_controls': NullControls(fine, so.get('null_controls')),
        'thermostat': Thermostat(records + replicas, float(so.get('temperature_rel_tol', 0.02))),
    }
    v['resolution'] = Resolution(fine, v['_seed_diff'], v['_refine_diff'], v['resolve_factor'])

    WriteVerificationMarkdown(v, fine, os.path.join(out_dir, 'verification.md'))
    export = {k: val for k, val in v.items() if not k.startswith('_')}
    export['seed_difference'] = v['_seed_diff']
    export['refinement_increment'] = v['_refine_diff']
    with open(os.path.join(out_dir, 'verification.yml'), 'w') as f:
        yaml.safe_dump(_clean(export), f, sort_keys=False, width=120)
    print(f"[INFO] verification written to {out_dir}/verification.md")

    if plots:
        others = {}
        if seed_sets:
            others[f"seed {replicas[0]['seed']}"] = seed_sets[0]
        if coarser:
            others[f"p = {records[-2]['level']}"] = coarser
        PlotParity(fine, others, v['surrogate_mc'], os.path.join(out_dir, 'plots'))
        from PlotSobolConvergence import plotSobolConvergence
        plotSobolConvergence(out_dir)
    return out_dir, v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Second-order Sobol study of the MD model with verification")
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "second_order.yml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="PARAM=VALUE", help="override parameter defaults")
    ap.add_argument("--levels", nargs="*", type=int, default=None, help="override the refinement levels")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--analyse_only", default=None, metavar="OUT_DIR",
                    help="re-analyse the campaigns of a previous study instead of running them")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    RunSecondOrder(args.config, args.set, args.levels, args.workers,
                   args.analyse_only or args.out_dir, reuse=bool(args.analyse_only), plots=not args.no_plots)
