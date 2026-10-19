"""
Utility functions for managing cache directories and configurations.

This module provides tools for inspecting, managing, and cleaning up
cache directories created by CachedWindowedDataset.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _get_cache_files_info(cache_dir: Path) -> tuple:
    """
    Helper function to get cache files and their total size.

    Args:
        cache_dir: Directory to scan for cache files.

    Returns:
        Tuple of (cache_files list, total_size in bytes)
    """
    cache_files = list(cache_dir.rglob("cache_*.pt"))
    total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
    return cache_files, total_size


def list_all_cache_configs(base_cache_dir: Union[str, Path] = "/local/cache/tokens") -> List[Dict[str, any]]:
    """
    List all cache configurations in the base cache directory.

    Args:
        base_cache_dir: Base directory containing cache subdirectories.
                       Defaults to /local/cache/tokens.


    Returns:
        List of dicts, each containing:
            - config_hash: The configuration hash (directory name)
            - cache_dir: Full path to the cache directory
            - num_files: Number of cache files
            - total_size_mb: Total size in MB
            - total_size_gb: Total size in GB
    """
    base_cache_dir = Path(base_cache_dir)

    if not base_cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {base_cache_dir}")
        return []

    configs = []
    for config_dir in base_cache_dir.iterdir():
        if config_dir.is_dir():
            # Count cache files and size
            cache_files, total_size = _get_cache_files_info(config_dir)

            configs.append({
                "config_hash": config_dir.name,
                "cache_dir": str(config_dir),
                "num_files": len(cache_files),
                "total_size_mb": total_size / 1e6,
                "total_size_gb": total_size / 1e9
            })

    # Sort by size (largest first)
    configs.sort(key=lambda x: x["total_size_mb"], reverse=True)

    return configs


def print_cache_summary(base_cache_dir: Union[str, Path] = "/local/cache/tokens"):
    """
    Print a human-readable summary of all cache configurations.

    Args:
        base_cache_dir: Base directory containing cache subdirectories.
    """
    configs = list_all_cache_configs(base_cache_dir)

    if not configs:
        print(f"No cache configurations found in {base_cache_dir}")
        return

    total_size_gb = sum(c["total_size_gb"] for c in configs)
    total_files = sum(c["num_files"] for c in configs)

    print(f"\nCache Summary for {base_cache_dir}")
    print("=" * 80)
    print(f"Total configurations: {len(configs)}")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print("\nConfigurations:")
    print("-" * 80)

    for config in configs:
        print(f"  Config Hash: {config['config_hash']}")
        print(f"    Files: {config['num_files']}")
        print(f"    Size: {config['total_size_mb']:.1f} MB ({config['total_size_gb']:.3f} GB)")
        print(f"    Path: {config['cache_dir']}")
        print()


def clear_cache_config(cache_dir: Union[str, Path], dry_run: bool = False) -> bool:
    """
    Remove a specific cache configuration directory.

    Args:
        cache_dir: Path to the cache directory to remove.
        dry_run: If True, only print what would be deleted without actually deleting.

    Returns:
        True if successful, False otherwise.
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return False

    if not cache_dir.is_dir():
        logger.error(f"Path is not a directory: {cache_dir}")
        return False

    # Safety check - ensure it looks like a cache directory
    cache_files, total_size = _get_cache_files_info(cache_dir)
    if not cache_files:
        logger.warning(f"Directory does not contain cache files: {cache_dir}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would remove: {cache_dir}")
        print(f"[DRY RUN]   Files: {len(cache_files)}")
        print(f"[DRY RUN]   Size: {total_size / 1e6:.1f} MB")
        return True

    try:
        shutil.rmtree(cache_dir)
        logger.info(f"Removed cache directory: {cache_dir} ({total_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Failed to remove cache directory {cache_dir}: {e}")
        return False


def clear_old_caches(
    base_cache_dir: Union[str, Path] = "/local/cache/tokens",
    keep_latest_n: int = 1,
    dry_run: bool = False
) -> int:
    """
    Remove old cache configurations, keeping only the most recent N.

    Keeps the N largest cache directories by size, assuming they are the most recent
    and actively used configurations.

    Args:
        base_cache_dir: Base directory containing cache subdirectories.
        keep_latest_n: Number of cache configurations to keep.
        dry_run: If True, only print what would be deleted without actually deleting.

    Returns:
        Number of cache directories removed.
    """
    configs = list_all_cache_configs(base_cache_dir)

    if len(configs) <= keep_latest_n:
        logger.info(f"Found {len(configs)} cache configs, keeping {keep_latest_n}. Nothing to remove.")
        return 0

    # Keep the largest N (assumed to be most recent/active)
    configs_to_keep = configs[:keep_latest_n]
    configs_to_remove = configs[keep_latest_n:]

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(configs_to_remove)} cache configs:")
        for config in configs_to_remove:
            print(f"  - {config['config_hash']} ({config['total_size_mb']:.1f} MB)")
        print(f"\n[DRY RUN] Would keep {len(configs_to_keep)} cache configs:")
        for config in configs_to_keep:
            print(f"  - {config['config_hash']} ({config['total_size_mb']:.1f} MB)")
        return len(configs_to_remove)

    removed_count = 0
    for config in configs_to_remove:
        if clear_cache_config(config["cache_dir"], dry_run=False):
            removed_count += 1

    logger.info(f"Removed {removed_count} old cache configurations")
    return removed_count


def get_cache_size(cache_dir: Union[str, Path]) -> Dict[str, float]:
    """
    Get the total size of a cache directory.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        Dict with keys 'bytes', 'mb', 'gb' containing size in different units.
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        return {"bytes": 0, "mb": 0, "gb": 0}

    _, total_bytes = _get_cache_files_info(cache_dir)

    return {
        "bytes": total_bytes,
        "mb": total_bytes / 1e6,
        "gb": total_bytes / 1e9
    }
