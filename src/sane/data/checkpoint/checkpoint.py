import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Checkpoint():
    model: nn.Module
    metadata: Optional[dict[str, Any]] = None
    zoo_root_dir: Optional[str] = None