import numpy as np
import random

from torchvision import datasets


class TripletFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.randint(0, len(pos_index) - 1)
        return self.samples[pos_index[rand]][0]

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)
        neg_path, neg_target = self._get_neg_sample(target)

        sample = self.loader(path)
        pos = self.loader(pos_path)
        neg = self.loader(neg_path)

        if self.transform is not None:
            sample = self.transform(sample)
            pos = self.transform(pos)
            neg = self.transform(neg)

        '''
        if self.target_transform is not None:
            target = self.target_transform(target)
            neg_target = self.target_transform(neg_target)
        '''

        c, h, w = pos.shape
        return sample, pos, neg, target
