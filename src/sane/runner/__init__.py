from sane.runner.base import Runner, RunResult
from sane.runner.local_runner import LocalRunner


# Lazy import: avoid pulling in the `ray` dependency until RayRunner is accessed.
def __getattr__(name: str):
    if name == "RayRunner":
        from sane.runner.ray_runner import RayRunner

        return RayRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Runner", "RunResult", "LocalRunner", "RayRunner"]