"""Abstract base class for learners that produce implementations from recorded data."""

import os
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from secretagent import config, savefile
from secretagent.dataset import Case, Dataset


class Learner(ABC):
    """Base class for learners that distill recorded interface calls into code.

    The constructor collects training data from recording directories,
    saves provenance info to a working directory, and loads the Dataset.
    Subclasses implement fit(), save_code(), and report().
    """

    def __init__(
        self,
        interface_name: str,
        dirs: list[Path],
        train_dir: str,
        file_under: str,
        latest: int = 1,
        check: Optional[list[str]] = None,
    ):
        self.interface_name = interface_name
        # filter recording dirs
        filtered_dirs = savefile.filter_paths(dirs, latest=latest, dotlist=check or [])
        # collect data and set up working directory
        config.configure(cfg={'train_dir': train_dir})
        self.out_dir, self.dataset = _collect_interface_data(
            filtered_dirs, interface_name, file_under=file_under)

    @abstractmethod
    def fit(self):
        """Fit the learner to the collected dataset."""
        ...

    @abstractmethod
    def save_code(self) -> Path:
        """Write learned implementation to self.out_dir. Return path to the file."""
        ...

    @abstractmethod
    def report(self) -> str:
        """Return a brief human-readable report on the learner."""
        ...


# --- data collection helpers (moved from utils.py) ---


def _collect_interface_data(
        dirs: list[Path], interface_name: str, file_under: str) -> tuple[Path, Dataset]:
    """Collect input/output pairs for an interface from recording directories.

    Creates a timestamped directory under config.get('train_dir') containing:
        data.json       — a JSON-serialized Dataset of input/output pairs
        sources.txt     — one source directory name per line
        source_configs/ — a copy of each source directory's config.yaml
    """
    train_dir = config.require('train_dir')
    dataset_filename, sources_filename, source_cfg_dirname = savefile.filename_list(
        train_dir, ['data.json', 'sources.txt', 'source_configs'], file_under)

    cases = _extract_cases_from_dirs(dirs, interface_name)
    dataset = Dataset(name=interface_name, cases=cases)
    with open(dataset_filename, 'w') as f:
        f.write(dataset.model_dump_json(indent=2))

    with open(sources_filename, 'w') as f:
        for d in dirs:
            f.write(f'{d}\n')

    os.makedirs(source_cfg_dirname, exist_ok=True)
    for d in dirs:
        src_cfg = Path(d) / 'config.yaml'
        if not src_cfg.exists():
            raise ValueError(f'missing config file {src_cfg}')
        shutil.copy2(src_cfg, source_cfg_dirname / f'{d.name}.yaml')

    return dataset_filename.parent, dataset


def _extract_cases_from_record(dx, lx, interface_name, record):
    """Yield Cases for the named interface from a single JSONL record."""
    for sx, step in enumerate(record.get('rollout', [])):
        if step['func'] == interface_name:
            yield Case(
                name=f'{interface_name}_{dx}.{lx}.{sx}',
                input_args=step.get('args'),
                input_kw=step.get('kw') or None,
                expected_output=step.get('output')
            )


def _extract_cases_from_dirs(dirs, interface_name):
    """Extract Cases for the named interface from results.jsonl in each directory."""
    result = []
    for dx, d in enumerate(dirs):
        jsonl_path = Path(d) / 'results.jsonl'
        if not jsonl_path.exists():
            raise ValueError(f'missing jsonl file {jsonl_path}')
        with open(jsonl_path) as f:
            for lx, line in enumerate(f):
                record = json.loads(line)
                for case in _extract_cases_from_record(dx, lx, interface_name, record):
                    result.append(case)
    return result
