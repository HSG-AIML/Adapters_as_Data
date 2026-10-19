from torch.utils.data import Dataset
from typing import List

from sane.data.checkpoint import Checkpoint

class CheckpointsDataset(Dataset[Checkpoint]):
    """The base dataset class for datasets of checkpoints.

    This base class handles a list of checkpoints, where each checkpoint is an instance of the `Checkpoint` class."""

    def __init__(self, checkpoints: List[Checkpoint] | None = None) -> None:
        """
        Args:
            checkpoints (Optional[List[Checkpoint]]): A list of checkpoints to initialize the dataset with. Defaults to None
        """
        super(CheckpointsDataset, self).__init__()
        self.name = self.__class__.__name__
        self.checkpoints = checkpoints

    def __len__(self) -> int:
        return len(self.checkpoints) if self.checkpoints else 0

    def __getitem__(self, index: int) -> Checkpoint:
        if self.checkpoints is None:
            raise IndexError("Dataset is empty.")
        return self.checkpoints[index]
