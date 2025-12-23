#!/usr/bin/env python3
"""
Scaffold a new Kaggle project directory inside this monorepo.

Usage:
  python scripts/new_project.py <project_name>
  python scripts/new_project.py <project_name> --title "Optional Title"

Example:
  python scripts/new_project.py playground-series-s5e13 --title "Playground Series S5E13"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Template:
    project_readme: str
    data_readme: str
    notebooks_readme: str
    weekly_report: str
    requirements: str


def render_templates(project_name: str, title: str) -> Template:
    today = date.today().isoformat()
    project_readme = f"""# {title}

本项目是本仓库（Kaggle 项目合集）中的一个子项目。

## Layout

- `data/`：数据下载说明（默认不提交原始数据）
- `notebooks/`：Kaggle kernel / notebook 入口脚本
- `src/`：训练与特征工程代码
- `reports/`：周报与实验记录（建议优先提交 Markdown）
"""

    data_readme = f"""# Data

默认不把原始数据提交到 GitHub（请确保数据放在 `data/raw/`，并被 `.gitignore` 忽略）。

## Expected files

请把数据放到：`{project_name}/data/raw/`

## Download

（在这里补充 Kaggle CLI 下载命令或数据来源链接）
"""

    notebooks_readme = """# Notebooks / Kernels

放 Kaggle kernel 的入口脚本与 metadata（如 `.ipynb` / `.py` / `kernel-metadata.json`）。
"""

    weekly_report = f"""# 周报｜{title}

日期：{today}

## 1. 本周目标

- TBD

## 2. 本周完成

- TBD

## 3. 实验与结果

- TBD（建议写：CV AUC/Logloss + Kaggle Public/Private LB）

## 4. 风险与注意事项

- TBD

## 5. 下周计划

- TBD
"""

    requirements = """numpy
pandas
scikit-learn
"""

    return Template(
        project_readme=project_readme,
        data_readme=data_readme,
        notebooks_readme=notebooks_readme,
        weekly_report=weekly_report,
        requirements=requirements,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name", help="Directory name, e.g. playground-series-s5e12")
    parser.add_argument("--title", default=None, help="Display title for README/weekly report")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    project_name = args.project_name.strip().strip("/").replace("\\", "/")
    if not project_name or project_name in {".", ".."} or "/" in project_name:
        raise SystemExit("project_name must be a simple directory name (no slashes).")

    project_dir = repo_root / project_name
    if project_dir.exists():
        raise SystemExit(f"Target already exists: {project_dir}")

    title = args.title or project_name
    tpl = render_templates(project_name=project_name, title=title)

    (project_dir / "data").mkdir(parents=True, exist_ok=False)
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=False)
    (project_dir / "notebooks").mkdir(parents=True, exist_ok=False)
    (project_dir / "src").mkdir(parents=True, exist_ok=False)
    (project_dir / "reports").mkdir(parents=True, exist_ok=False)

    (project_dir / "README.md").write_text(tpl.project_readme, encoding="utf-8")
    (project_dir / "data" / "README.md").write_text(tpl.data_readme, encoding="utf-8")
    (project_dir / "notebooks" / "README.md").write_text(tpl.notebooks_readme, encoding="utf-8")
    (project_dir / "reports" / "weekly_report.md").write_text(tpl.weekly_report, encoding="utf-8")
    (project_dir / "requirements.txt").write_text(tpl.requirements, encoding="utf-8")

    print(f"Created: {project_dir}")
    print("Next:")
    print(f"- Add your code under: {project_name}/src")
    print(f"- Put Kaggle kernels under: {project_name}/notebooks")
    print(f"- Keep data under: {project_name}/data/raw (ignored by git)")


if __name__ == "__main__":
    main()

