"""Method constants for quick modelEvaluation（来自 config.toml [reports_generation.quick_eval]）。"""

from core.config import default_quick_eval_config, getReportsGenerationConfig

try:
    _qe = getReportsGenerationConfig().get("quick_eval")
    if not isinstance(_qe, dict):
        raise TypeError("quick_eval 不是 dict")
    for _k in ("basic_methods", "optimized_methods", "all_methods"):
        if _k not in _qe:
            raise KeyError(_k)
except Exception:
    _qe = default_quick_eval_config()

BASIC_METHODS = list(_qe["basic_methods"])
OPTIMIZED_METHODS = list(_qe["optimized_methods"])
ALL_METHODS = list(_qe["all_methods"])
