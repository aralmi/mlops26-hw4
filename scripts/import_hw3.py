"""Восстановить точный снимок входа ДЗ 4 из локальной папки ДЗ 3."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.config import load_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("../../mlops26-hw3/hw3-broken/data"))
    args = parser.parse_args()
    manifest = json.loads(Path("docs/input_manifest.json").read_text(encoding="utf-8"))
    params = load_params()
    destinations = {
        "train.jsonl": Path(params["data"]["train_jsonl"]),
        "val.jsonl": Path(params["data"]["val_jsonl"]),
        "attribution.jsonl": Path(params["data"]["train_jsonl"]).parent / "attribution.jsonl",
    }
    # Сначала проверяем ВСЕ файлы. Отличающийся снимок нельзя подменять молча.
    for name, meta in manifest["files"].items():
        source = args.source_dir / name
        if not source.is_file():
            raise SystemExit(f"Нет {source}; восстановите данные ДЗ 3 через DVC.")
        if hashlib.sha256(source.read_bytes()).hexdigest() != meta["sha256"]:
            raise SystemExit(f"{source}: SHA256 не совпадает с входным снимком ДЗ 4")
        target = destinations[name]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != meta["sha256"]:
            raise SystemExit(f"{target}: уже есть другие данные, импорт остановлен без перезаписи")
    for name, target in destinations.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(args.source_dir / name, target)
        print(f"✓ {target}: SHA256 совпадает с ДЗ 3")


if __name__ == "__main__":
    main()
