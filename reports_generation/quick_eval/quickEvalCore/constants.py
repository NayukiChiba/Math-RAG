"""Method constants for quick modelEvaluation（来自 config.toml [reports_generation.quick_eval]）。"""

from core.config import getReportsGenerationConfig

_qe = getReportsGenerationConfig()["quick_eval"]
BASIC_METHODS = list(_qe["basic_methods"])
OPTIMIZED_METHODS = list(_qe["optimized_methods"])
ALL_METHODS = list(_qe["all_methods"])
