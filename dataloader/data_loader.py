import os
import sys
from torch.utils import data
import pickle

class AllGraphDataSampler(data.Dataset):
    def __init__(self, base_dir, gname_list=None,
                 data_start=None, data_middle=None, data_end=None,
                 train_start_date=None, train_end_date=None,
                 val_start_date=None, val_end_date=None,
                 test_start_date=None, test_end_date=None,
                 idx=False, date=True,
                 mode="train"):
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        if gname_list is None:
            self.gnames_all = os.listdir(self.data_dir)
            self.gnames_all.sort()
        if idx:
            if mode == "train":
                self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
            elif mode == "val":
                self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
            elif mode == "test":
                self.gnames_all = self.gnames_all[self.data_end:]
        if date:
            if mode == "train":
                lo, hi = train_start_date, train_end_date
            elif mode == "val":
                lo, hi = val_start_date, val_end_date
            else:
                lo, hi = test_start_date, test_end_date
            self.gnames_all = self.gnames_all[
                self.date_to_idx(lo, "start"):self.date_to_idx(hi, "end") + 1]
        self.data_all = self.load_state()

    def __len__(self):
        return len(self.data_all)

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        skipped = 0
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            d = pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb"))
            # skip degenerate samples (early days without a full lookback window)
            lbl = d.get('labels')
            if lbl is None or (hasattr(lbl, 'shape') and lbl.shape[0] == 0):
                skipped += 1
                continue
            data_all.append(d)
        print('{} data loaded! ({} samples, {} empty skipped)'.format(
            self.mode, len(data_all), skipped))
        return data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def date_to_idx(self, date, which="start"):
        """Map a split-boundary date to a file index.

        which="start": first trading day on/after `date`.
        which="end":   last trading day on/before `date`.
        This keeps train/val/test splits from leaking when a boundary date
        is not itself a trading day.
        """
        dates = [g[:10] for g in self.gnames_all]
        if date in dates:
            return dates.index(date)
        if which == "start":
            for i, d in enumerate(dates):
                if d >= date:
                    return i
            return len(dates) - 1
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= date:
                return i
        return 0
