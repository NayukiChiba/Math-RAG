"""统一文件加载入口。"""

import json
import pickle
from typing import Any


class FileLoader:
    """统一封装常见数据加载逻辑。"""

    def toml(self, path: str) -> dict[str, Any]:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        with open(path, "rb") as f:
            return tomllib.load(f)

    def json(self, path: str) -> Any:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def jsonl(self, path: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def pickle(self, path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def text_lines(self, path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]


_DEFAULT_FILE_LOADER = FileLoader()


def getFileLoader() -> FileLoader:
    """获取可复用的全局 FileLoader 实例。"""
    return _DEFAULT_FILE_LOADER
