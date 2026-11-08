"""motorsim: motor modelling, closed-loop simulation, identification and learning.

    core/      motor-independent engine (config, models, control, sim, analysis, io, plot, learning)
    domains/   motor types (dc_motor, pmsm) plugged into the engine
    pipeline/  experiments, identification workflow, CLI commands
"""
from motorsim.core.config import flat_params, load_config, sample_params, with_params
from motorsim.core.io import load_result
from motorsim.core.model.results import SimResult
from motorsim.core.sim import Motor, simulate

__version__ = "0.2.0"