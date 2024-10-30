import numpy as np
import random
from torch.utils.data import Sampler

class RandomSampler(Sampler):
    def __init__(self, dataset, task_type, n_classes, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.task_type = task_type
        self.n_classes = n_classes

        # Process labels based on task type (gender or dialect)
        self.labels = [self.get_label(label) for label in dataset.label_list]
        self.class_to_index = [[] for _ in range(self.n_classes)]
        for idx, lbl in enumerate(self.labels):
            self.class_to_index[lbl].append(idx)

        self.num_iter = (
            (self.n_classes + 1) * max([len(x) for x in self.class_to_index]) - 1) // self.batch_size + 1

    def __len__(self):
        return self.num_iter

    def __repr__(self):
        return self.__str__()

    def _prepare(self):
        for x in self.class_to_index:
            random.shuffle(x)

        max_len = max([len(x) for x in self.class_to_index])
        len_class_1 = len(self.class_to_index[1])
        index = []
        list_class = list(range(self.n_classes))
        for i in range(max_len):
            random.shuffle(list_class)
            for c in list_class:
                index.append(
                    self.class_to_index[c][i % len(self.class_to_index[c])])
            index.append(
                self.class_to_index[1][(len_class_1 - i) % len_class_1])

        return np.array(index, dtype=np.longlong)

    def get_label(self, original_label):
        """Determine the label based on the task type."""
        if self.task_type == "gender":
            return int(original_label // 3)
        elif self.task_type == "dialect":
            return int(original_label % 3)

    def __iter__(self):
        indices = self._prepare()
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]