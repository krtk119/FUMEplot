"""
Build, run and analyse an EasyVVUQ sensitivity campaign for the MD application
and export the Sobol indices in the `sobols.yml` layout read by FUMEplot.

    python3 run_campaign.py [campaign.yml] [--workers N] [--set n_steps=600 ...]
    python3 run_campaign.py [campaign.yml] --analyse_only <campaign_dir>

The exported file extends the layout of fumeplot/PlotSobolByTimestep.py:

    campaign_info:                     # sampling design, made explicit
        sampler, uncertain_params, fixed_params, application, ...
        qoi_layout: {qoi: {shape: [...], axes: {name: [...]}}}
    qoi_statistics:
        <qoi>: {mean: [...], std: [...]}
    <param>:
        <qoi>_sobols_first: [...]      # flattened, row-major
        <qoi>_sobols_total: [...]
        <qoi>_sobols_second: {<other_param>: [...]}
"""

import os
import sys
import argparse
import datetime
import functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
import chaospy as cp
import easyvvuq as uq
from easyvvuq.actions import Actions, CreateRunDirectory, Encode, Decode

from md_wrappers import MDEncoder, MDDecoder, MDRunner, ReadQoiLayout

HERE = os.path.dirname(os.path.abspath(__file__))

# one BLAS/OpenMP thread per MD run, otherwise parallel runs oversubscribe the cores
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

class QMCSamplerSeeded(uq.sampling.QMCSampler, sampler_name="qmc_sampler_seeded"):
    """
    QMCSampler with a reproducible design: EasyVVUQ 1.2.3 draws a scrambled Sobol
    sequence (SALib) without a seed, so every campaign gets a different design.
    """

    def __init__(self, vary, n_mc_samples, seed=None, count=0):
        import easyvvuq.sampling.qmc as qmc
        original = qmc.sobol.sample
        qmc.sobol.sample = functools.partial(original, seed=seed)
        try:
            super().__init__(vary, n_mc_samples, count=count)
        finally:
            qmc.sobol.sample = original


SAMPLERS = {'QMCSampler': QMCSamplerSeeded}


class QMCAnalysisBootstrapFix(uq.analysis.QMCAnalysis):
    """
    EasyVVUQ 1.2.3: when n_mc * n_bootstrap * n_qoi > 1e7, sobol_bootstrap takes a
    sequential branch that indexes the resampling matrix as r[i] instead of
    r[:, i] and fails. Cap n_bootstrap to stay in the (correct) vectorised branch;
    this only affects the confidence intervals, not the index estimates.
    """

    def sobol_bootstrap(self, samples, alpha=0.05, n_bootstrap=1000):
        n_mc = len(samples) // (self.sampler.n_params + 2)
        n_qoi = np.asarray(samples[0]).size
        n_bootstrap = max(1, min(n_bootstrap, 10**7 // (n_mc * n_qoi)))
        return super().sobol_bootstrap(samples, alpha=alpha, n_bootstrap=n_bootstrap)


ANALYSIS_FOR_SAMPLER = {
    'PCESampler': uq.analysis.PCEAnalysis,
    'SCSampler': uq.analysis.SCAnalysis,
    'QMCSampler': QMCAnalysisBootstrapFix,
}


def LoadConfig(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split('=', 1)
        if key not in cfg['parameters']:
            sys.exit(f"[ERROR] --set {key}: not a parameter in {path}")
        cfg['parameters'][key]['default'] = yaml.safe_load(value)
    return cfg


def AllQois(cfg):
    return [q for group in cfg['qois'].values() for q in group]


def BuildCampaign(cfg, cfg_dir):
    app = cfg['application']
    work_dir = os.path.join(cfg_dir, cfg['campaign']['work_dir'])
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    params = {k: {'type': v['type'], 'default': v['default']} for k, v in cfg['parameters'].items()}
    vary = {k: getattr(cp, v['dist'])(*v['args']) for k, v in cfg['uncertain'].items()}

    # 'md' adds the MD consistency checks; 'generic' is a plain template encoder (e.g. test functions)
    encoder_cls = uq.encoders.GenericEncoder if app.get('encoder') == 'generic' else MDEncoder
    encoder = encoder_cls(template_fname=os.path.join(cfg_dir, app['template']), delimiter='$',
                          target_filename=app['input_file'])
    decoder = MDDecoder(target_filename=app['output_file'], output_columns=AllQois(cfg))
    actions = Actions(CreateRunDirectory(work_dir, flatten=True),
                      Encode(encoder),
                      MDRunner(app['runner'], app['input_file'], app['output_file'],
                               app_dir=os.path.join(cfg_dir, app.get('app_dir', 'md_app'))),
                      Decode(decoder))

    campaign = uq.Campaign(name=cfg['campaign']['name'], work_dir=work_dir,
                           params=params, actions=actions)
    sampler_cls = SAMPLERS.get(cfg['sampler']['type']) or getattr(uq.sampling, cfg['sampler']['type'])
    campaign.set_sampler(sampler_cls(vary=vary, **cfg['sampler'].get('kwargs', {})))
    return campaign


def ReloadCampaign(cfg, campaign_dir):
    db = os.path.join(os.path.abspath(campaign_dir), 'campaign.db')
    if not os.path.isfile(db):
        sys.exit(f"[ERROR] no campaign.db in {campaign_dir}")
    return uq.Campaign(name=cfg['campaign']['name'], db_location=f"sqlite:///{db}")


def Analyse(campaign, cfg):
    analysis_cls = ANALYSIS_FOR_SAMPLER[cfg['sampler']['type']]
    analysis = analysis_cls(sampler=campaign.get_active_sampler(), qoi_cols=AllQois(cfg),
                            **cfg['sampler'].get('analysis_kwargs', {}))
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()
    results.analysis_element = analysis      # needed for the SC full decomposition below
    return results


def SobolsFromDecomposition(results, qoi, params):
    """
    First, second and total order from the full ANOVA decomposition {u: S_u}
    (SCAnalysis reports only first order directly): S_i = S_(i), S_ij = S_(i,j),
    S_Ti = sum of S_u over all u containing i.
    """
    analysis = getattr(results, 'analysis_element', None)
    if not isinstance(analysis, uq.analysis.SCAnalysis):
        return None
    if analysis.sparse:
        decomposition = analysis.get_pce_sobol_indices(qoi, 'all')[3]
    else:
        decomposition = analysis.get_sobol_indices(qoi, 'all')
    first = {p: decomposition[(i,)] for i, p in enumerate(params)}
    total = {p: sum(v for u, v in decomposition.items() if i in u) for i, p in enumerate(params)}
    second = {p: {q: decomposition[tuple(sorted((i, j)))] for j, q in enumerate(params) if j != i}
              for i, p in enumerate(params)}
    return first, total, second


def _toList(a, size=None):
    """Flatten; a size-1 value is broadcast to `size` (degenerate designs give scalars)."""
    a = np.asarray(a, dtype=float).ravel()
    if size is not None and a.size == 1 and size > 1:
        a = np.full(size, a[0])
    return a.tolist()


def ExportSobols(results, campaign, cfg, out_path):
    """Write Sobol indices + the sampling design to a FUMEplot sobols.yml."""
    qois = AllQois(cfg)
    params = list(cfg['uncertain'])
    runs = list(campaign.list_runs())
    run_dir = runs[0][1]['run_dir']

    info = {
        'name': cfg['campaign']['name'],
        'campaign_dir': campaign.campaign_dir,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'easyvvuq_version': uq.__version__,
        'measure': 'sobol',
        'application': cfg['application'],
        'sampler': {**cfg['sampler'], 'n_samples': len(runs)},
        'analysis': ANALYSIS_FOR_SAMPLER[cfg['sampler']['type']].__name__,
        'param_order': params,
        'uncertain_params': cfg['uncertain'],
        'fixed_params': {k: v['default'] for k, v in cfg['parameters'].items() if k not in cfg['uncertain']},
        'qoi_groups': cfg['qois'],
        'labels': cfg.get('labels', {}),
        'qoi_layout': ReadQoiLayout(run_dir, cfg['application']['output_file'], qois),
    }

    report = {'campaign_info': info, 'qoi_statistics': {}}
    report.update({p: {} for p in params})

    for qoi in qois:
        n = int(np.prod(info['qoi_layout'][qoi]['shape']))

        def flat(a):
            return _toList(a, n)

        report['qoi_statistics'][qoi] = {s: flat(results.describe(qoi, s)) for s in ('mean', 'std')}
        decomposed = SobolsFromDecomposition(results, qoi, params)
        if decomposed:
            first, total, second_all = decomposed
        else:
            first, total, second_all = results.sobols_first(qoi), results.sobols_total(qoi), None
        for p in params:
            report[p][f"{qoi}_sobols_first"] = flat(first[p])
            report[p][f"{qoi}_sobols_total"] = flat(total[p])
            try:
                second = second_all[p] if second_all else results.sobols_second(qoi, p)
                report[p][f"{qoi}_sobols_second"] = {o: flat(v) for o, v in second.items()}
            except (RuntimeError, NotImplementedError, KeyError, TypeError) as e:
                if p == params[0]:
                    print(f"[INFO] no second-order Sobol indices for {qoi}: {type(e).__name__}")

    with open(out_path, 'w') as f:
        yaml.safe_dump(report, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"[INFO] Sobol indices written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EasyVVUQ Sobol campaign for the MD application")
    ap.add_argument("config", nargs="?", default=os.path.join(HERE, "campaign.yml"))
    ap.add_argument("--workers", type=int, default=None, help="parallel runs (default: from config)")
    ap.add_argument("--set", nargs="*", default=[], metavar="PARAM=VALUE",
                    help="override parameter defaults, e.g. n_steps=600")
    ap.add_argument("--analyse_only", default=None, metavar="CAMPAIGN_DIR",
                    help="re-analyse an existing campaign instead of running a new one")
    args = ap.parse_args()

    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = LoadConfig(args.config, args.set)

    if args.analyse_only:
        campaign = ReloadCampaign(cfg, args.analyse_only)
    else:
        campaign = BuildCampaign(cfg, cfg_dir)
        n_workers = args.workers or cfg['execution']['n_workers']
        print(f"[INFO] {campaign.get_active_sampler().n_samples} runs on {n_workers} workers "
              f"in {campaign.campaign_dir}")
        with ThreadPoolExecutor(n_workers) as pool:   # runs are subprocesses; threads suffice
            campaign.execute(pool=pool).collate(progress_bar=True)

    results = Analyse(campaign, cfg)
    ExportSobols(results, campaign, cfg, os.path.join(campaign.campaign_dir, 'sobols.yml'))
