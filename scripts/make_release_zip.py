#!/usr/bin/env python3
"""Build CipherMind_Sentinel_Final.zip — clean release staging."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

SRC = Path("/home/z/my-project")
STAGE = Path("/tmp/CipherMind_Sentinel")
OUT = Path("/home/z/my-project/CipherMind_Sentinel_Final.zip")

# top-level files to copy verbatim
TOP_FILES = [
    ".env.example",
    "README.md",
    "requirements.txt",
    "package.json",
    "bun.lock",
    "next.config.ts",
    "tsconfig.json",
    "tailwind.config.ts",
    "postcss.config.mjs",
    "components.json",
    "eslint.config.mjs",
    "next-env.d.ts",
]

# full directories (copied recursively with excludes)
DIRS = [
    "src",
    "public",
    "prisma",
    "docs",
    "notebooks",
    "dataset",
    "mini-services/soc-engine",
]

EXCLUDE_NAMES = {"node_modules", ".next", "__pycache__", ".pytest_cache",
                 ".ipynb_checkpoints", ".git", "engine.log", "frontend.log"}

ML_INCLUDE = [  # ml/artifacts subtree (explicit, everything canonical)
    "models/binary_lightgbm.txt", "models/binary_lightgbm.json",
    "models/multiclass_lightgbm.txt", "models/multiclass_lightgbm.json",
    "models/isolation_forest.json",
    "preprocessor/feature_config.json", "preprocessor/clustering.json",
    "explainability/shap_cache.json",
    "replay/demo_sequence.json", "replay/boot_sample.json",
    "metrics/model_comparison.json", "metrics/feature_ablation.json",
    "metrics/calibration.json", "metrics/test_evaluation.json",
    "metrics/shap_global.json", "metrics/multiclass_gain.json",
    "metrics/operational.json", "metrics/dataset_profile.json",
    "metrics/train_prevalence.json",
    "reports/eval_summary.json", "reports/notebook_verification.json",
    "metadata/model_registry.json",
]

SCRIPTS_INCLUDE = ["start-all.sh", "stop-all.sh", "start-all.bat", "stop-all.bat"]
TESTS_INCLUDE = ["validate_ts_engine.py"]

GITIGNORE = """# dependencies
node_modules/
# next.js
.next/
out/
# env
.env
.env*.local
# logs
*.log
# python
__pycache__/
.ipynb_checkpoints/
.venv/
# databases (generated)
db/
*.db
# os/editor
.DS_Store
.idea/
.vscode/
"""


def copy_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*EXCLUDE_NAMES))
    return n


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    # top-level files
    for f in TOP_FILES:
        p = SRC / f
        if p.exists():
            shutil.copy2(p, STAGE / f)
        else:
            print(f"  (skip missing top file: {f})")

    # directories
    for d in DIRS:
        copy_tree(SRC / d, STAGE / d)
        print(f"  + {d}/")

    # ml artifacts (explicit canonical set)
    for rel in ML_INCLUDE:
        p = SRC / "ml" / "artifacts" / rel
        if not p.exists():
            raise FileNotFoundError(f"MISSING ARTIFACT: {rel}")
        dst = STAGE / "ml" / "artifacts" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    print("  + ml/artifacts/ (22 canonical artifacts)")

    # scripts + tests (explicit include — sandbox tooling stays out)
    (STAGE / "scripts").mkdir(exist_ok=True)
    for f in SCRIPTS_INCLUDE:
        shutil.copy2(SRC / "scripts" / f, STAGE / "scripts" / f)
    (STAGE / "tests").mkdir(exist_ok=True)
    for f in TESTS_INCLUDE:
        shutil.copy2(SRC / "tests" / f, STAGE / "tests" / f)
    print("  + scripts/ (start/stop) + tests/ (validate_ts_engine.py)")

    (STAGE / ".gitignore").write_text(GITIGNORE)

    # sanity checks before zipping
    assert (STAGE / "notebooks" / "CipherMind_Model_Training_and_Evaluation.ipynb").exists()
    assert (STAGE / "ml" / "artifacts" / "metadata" / "model_registry.json").exists()
    assert (STAGE / "src" / "app" / "page.tsx").exists()
    assert (STAGE / "mini-services" / "soc-engine" / "src" / "index.ts").exists()
    assert (STAGE / "dataset" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv").exists()
    assert not (STAGE / "node_modules").exists()
    assert not any((STAGE).rglob("node_modules"))

    # zip it
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(STAGE.parent))
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nZIP ready: {OUT}  ({size_mb:.1f} MB)")
    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
    print(f"files in zip: {len(names)}")
    # breakdown
    from collections import Counter
    top = Counter(n.split("/")[2] if n.startswith("CipherMind_Sentinel/") and n.count("/") >= 2 else n.split("/")[1] for n in names)
    for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:14]:
        print(f"   {k:28s} {v} files")


if __name__ == "__main__":
    main()
