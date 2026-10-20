"""
Assurance test: the pipeline computes and compares Sobol indices for the
pulled-cloud QoIs and the diffusion coefficient.

Runs a small real campaign (degree-1 PCE: 2^4 = 16 short MD runs), exports
sobols.yml, reloads it through the FUMEplot loader and checks, for

    pull_disp, cloud_spread, diff_coeff        (0D)
    pull_disp_t, cloud_spread_t                (1D)

that the indices exist, are valid (0 <= S_1 <= S_T <= 1 up to tolerance), form
a partition of unity, and that the comparison plots are produced.

    python3 test_sensitivity_qois.py        (or: pytest test_sensitivity_qois.py)
"""

import os
import sys
import glob
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fumeplot'))

from run_campaign import LoadConfig, BuildCampaign, Analyse, ExportSobols   # noqa: E402
from SobolPartition import LoadSobolSet, Partition                          # noqa: E402
from PlotSobolScalar import plotSobolScalar                                 # noqa: E402
from PlotSobolProfile import plotSobolProfile                               # noqa: E402

SCALAR_QOIS = ['pull_disp', 'cloud_spread', 'diff_coeff']
PROFILE_QOIS = ['pull_disp_t', 'cloud_spread_t']
TOL = 1e-6


def runSmallCampaign(work_dir):
    cfg = LoadConfig(os.path.join(HERE, 'campaign.yml'),
                     ['n_steps=400', 'n_equil=100', 'rdf_window=50'])
    cfg['campaign']['work_dir'] = work_dir
    cfg['sampler']['kwargs']['polynomial_order'] = 1
    cfg['qois'] = {'scalar': SCALAR_QOIS, 'profile': PROFILE_QOIS}

    campaign = BuildCampaign(cfg, HERE)
    with ThreadPoolExecutor(8) as pool:
        campaign.execute(pool=pool).collate()
    results = Analyse(campaign, cfg)
    out = os.path.join(campaign.campaign_dir, 'sobols.yml')
    ExportSobols(results, campaign, cfg, out)
    return out


def checkQoi(sset, name, ndim):
    assert name in sset.qois, f"{name}: missing from sobols.yml"
    qoi = sset.qois[name]
    assert qoi.ndim == ndim, f"{name}: expected {ndim}D, got shape {qoi.shape}"

    comps, mask = Partition(sset, qoi)
    assert not mask.all(), f"{name}: zero variance everywhere"
    valid = ~mask
    for p in sset.params:
        s1, st = qoi.first[p][valid], qoi.total[p][valid]
        assert np.all(np.isfinite(s1)) and np.all(np.isfinite(st)), f"{name}/{p}: non-finite index"
        assert np.all(s1 >= -TOL) and np.all(st <= 1 + TOL), f"{name}/{p}: index outside [0, 1]"
        assert np.all(st >= s1 - TOL), f"{name}/{p}: total < first order"

    total = np.nansum([c.values for c in comps], axis=0)
    assert np.allclose(total[valid], 1.0, atol=1e-6), f"{name}: partition does not sum to 1"
    return {p: float(np.nanmean(qoi.total[p][valid])) for p in sset.params}


def test_sensitivity_qois():
    with tempfile.TemporaryDirectory() as tmp:
        sset = LoadSobolSet(runSmallCampaign(tmp))

        summary = {q: checkQoi(sset, q, 0) for q in SCALAR_QOIS}
        summary.update({q: checkQoi(sset, q, 1) for q in PROFILE_QOIS})
        assert sset.qois['pull_disp'].mean > 0, "pulled cloud centroid should move along +x"

        plotSobolScalar('md', sset, plot_path=tmp, qois=SCALAR_QOIS)
        plotSobolProfile('md', sset, plot_path=tmp, qois=PROFILE_QOIS)
        pngs = {os.path.basename(f) for f in glob.glob(f"{tmp}/EnsemblePlots/mdPlots/Sobol/*.png")}
        for f in ['Sobol_scalar_partition.png', 'Sobol_scalar_first_total.png',
                  'pull_disp_t_Sobol_profile.png', 'cloud_spread_t_Sobol_profile.png',
                  'Sobol_compare_pull_disp_t_cloud_spread_t.png']:
            assert f in pngs, f"missing plot {f}"

    print("\nmean total-order index per input:")
    print(f"{'QoI':16s}" + ''.join(f"{p:>13s}" for p in sset.params))
    for q, row in summary.items():
        print(f"{q:16s}" + ''.join(f"{row[p]:13.3f}" for p in sset.params))


if __name__ == "__main__":
    test_sensitivity_qois()
    print("[PASS] Sobol indices computed and compared for", ', '.join(SCALAR_QOIS + PROFILE_QOIS))
