"""产品线 CLI 可恢复的用户级错误（由入口统一转为退出码）。"""


class CliUserError(Exception):
    """参数或环境不满足时的显式错误，避免在业务 helper 中直接 SystemExit。"""
