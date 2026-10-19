from abc import ABC, abstractmethod
from typing import List, Sequence
from torch.utils.data import random_split, Subset

from sane.data.datasets.checkpoints_dataset import CheckpointsDataset


class Splitter(ABC):
    """ The dataset splitter mutator base class. """
    def __call__(self, *args, **kwargs)-> List[Subset[CheckpointsDataset]]:
        return self.split(*args, **kwargs)

    @abstractmethod
    def split(self, dataset: CheckpointsDataset)-> List[Subset[CheckpointsDataset]]:
        pass

class RandomSplitter(Splitter):
    """ A mutator class for splitting a dataset using ratios.
    
    Using this splitter is, for most cases, a bad choice. Since the splitting is done at random, it can lead to leakage between train and test sets.
    For example, if the dataset contains checkpoints of the same model, the train and test sets may contain checkpoints of the same model.
    It can also be the case the multiple checkpoints trained with the same configuration but different seed are split into train and test sets, albeit not being significantly different.
    Use this mutator only if you are aware of the risks and limitations.

    Args:
        dataset (CheckpointsDataset): The dataset to split.
        ratios (Sequence[int | float]): The ratios to split the dataset into. The sum of the ratios must be equal to 1.
    """
    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        return random_split(dataset, ratios)

class SequentialSplitter(Splitter):
    """Split a dataset into contiguous blocks by ratio.

    The first `ratios[0]` fraction of indices (0, 1, 2, ...) goes to the
    first split, the next fraction to the second, and so on. No shuffling,
    so the result is fully deterministic and identical across runs and
    models, independent of any RNG state.

    Args:
        dataset: The dataset to split.
        ratios: Fractions per split; should sum to 1.0.
    """
    def split(self, dataset: CheckpointsDataset, ratios: Sequence[int | float]) -> List[Subset[CheckpointsDataset]]:
        n = len(dataset)

        # Convert ratios -> integer sizes, giving any rounding remainder
        # to the last split so the sizes always sum to n exactly.
        sizes = [int(r * n) for r in ratios]
        sizes[-1] = n - sum(sizes[:-1])

        subsets = []
        start = 0
        for size in sizes:
            subsets.append(Subset(dataset, list(range(start, start + size))))
            start += size
        return subsets