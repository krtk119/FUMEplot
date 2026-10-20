"""
EasyVVUQ wrappers for the MD application: encoder, decoder and runner.

    MDEncoder   template -> md_input.json, with a physical consistency check
    MDDecoder   md_output.json -> {qoi: scalar | flat vector}
    MDRunner    ExecuteLocal action launching the MD engine in the run directory

EasyVVUQ analyses treat every QoI as a scalar or a 1D vector, so 2D QoIs
(e.g. g(t, r)) are flattened row-major by the decoder. Their shape and
coordinates are read once per campaign with `ReadQoiLayout` and exported next
to the Sobol indices, so the plotting side can restore them.

To swap the toy engine for a production code (e.g. LAMMPS), replace the
template, the runner command in campaign.yml and the output parsing in
`MDDecoder._readRaw`; the campaign and plotting code stay unchanged.
"""

import os
import sys
import json

import numpy as np
import easyvvuq as uq
from easyvvuq import OutputType
from easyvvuq.actions import ExecuteLocal

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'md_app')


class MDEncoder(uq.encoders.GenericEncoder):
    """GenericEncoder that rejects parameter sets the MD engine cannot run."""

    def encode(self, params={}, target_dir=''):
        n_atoms = 4 * int(params['n_cells'])**3
        half_box = 0.5 * (n_atoms / float(params['density']))**(1.0 / 3.0)
        for key in ('r_cut', 'r_max'):
            if float(params[key]) > half_box:
                raise ValueError(f"{key}={params[key]} exceeds half the box ({half_box:.3f}) "
                                 f"at density={params['density']}; increase n_cells")
        super().encode(params=params, target_dir=target_dir)


class MDDecoder:
    """Reads the MD JSON output; scalars stay scalars, 1D/2D QoIs become flat lists."""

    def __init__(self, target_filename, output_columns):
        if len(output_columns) == 0:
            raise RuntimeError("output_columns cannot be empty.")
        self.target_filename = target_filename
        self.output_columns = output_columns
        self.output_type = OutputType('sample')

    def _readRaw(self, run_dir):
        with open(os.path.join(run_dir, self.target_filename)) as f:
            return json.load(f)

    def parse_sim_output(self, run_info={}):
        raw = self._readRaw(run_info['run_dir'])
        out = {}
        for qoi in self.output_columns:
            if qoi not in raw:
                raise RuntimeError(f"no such field: {qoi} in {self.target_filename}")
            value = np.asarray(raw[qoi], dtype=float)
            out[qoi] = float(value) if value.ndim == 0 else value.ravel().tolist()
        return out


def ReadQoiLayout(run_dir, output_file, qois):
    """{qoi: {'shape': [...], 'axes': {name: [...]}}} from one run's output."""
    with open(os.path.join(run_dir, output_file)) as f:
        raw = json.load(f)
    return {q: {'shape': list(np.shape(raw[q])),
                'axes': raw.get('axes', {}).get(q, {})}
            for q in qois}


def MDRunner(command, input_file, output_file, app_dir=APP_DIR):
    """
    `command` is a template from campaign.yml, e.g.
        "{python} {app_dir}/lj_md.py {input} {output}"
    It is executed with the run directory as working directory.
    """
    cmd = command.format(python=sys.executable, app_dir=app_dir,
                         input=input_file, output=output_file)
    return ExecuteLocal(cmd)
