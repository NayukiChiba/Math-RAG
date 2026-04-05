"""实验相关脚本集合。"""

from runners.experiments.experimentWebUI import main as launch_experiment_webui
from runners.experiments.runExperiments import main as run_experiments

__all__ = ["launch_experiment_webui", "run_experiments"]
