import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from sane.orchestrator import Orchestrator

_force_local = False


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)

    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    sane_logger = logging.getLogger("sane")
    sane_logger.setLevel(log_level)
    sane_logger.propagate = False
    if not sane_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        sane_logger.addHandler(handler)

    orchestrator = Orchestrator(config=config, force_local=_force_local)
    result = orchestrator.run()
    print(f"\nExperiment finished. Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        _force_local = True
    main()