import json
import psutil
import os
from typing import Dict, List, Optional, Union

def parse_json(path: str) -> Dict:
    """Parses a JSON file and return the content as a dictionary."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data
        

def parse_jsonl(path: str) -> List[Dict]:
    """Parses a JSONL file and return the content as a list of dictionaries."""
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def resolve_abs_path(path:str) -> str:
    """
    Args:
        path (str): The relative/absolute file path to validate.

    Returns:
        str: The absolute path if it exists.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")
    return abs_path

_last_rss = [None]

def log_mem(tag):
    rss = psutil.Process(os.getpid()).memory_info().rss
    rss_gb = rss / 1e9
    if _last_rss[0] is None:
        print(f"[MEM] {tag}: {rss_gb:.2f} GB", flush=True)
    else:
        delta_gb = (rss - _last_rss[0]) / 1e9
        sign = "+" if delta_gb >= 0 else ""
        print(f"[MEM] {tag}: {rss_gb:.2f} GB ({sign}{delta_gb:.2f} GB)", flush=True)
    _last_rss[0] = rss