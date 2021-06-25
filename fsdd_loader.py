import numpy as np
import scipy.io.wavfile as wav
import os


class DataStream:
    def __init__(self, what, batch=100, seed=1000, frac=1, root_dir=None):
        self.root_dir = (
            "data/free-spoken-digit-dataset/recordings/"
            if root_dir is None
            else root_dir
        )
        self.what = what
        self.last_idx = 0
        self.file_names = []
        self.batch_size = batch
        self.seed = seed

        for file_name in sorted(os.listdir(self.root_dir)):
            fs, raw_wav = wav.read(self.root_dir + file_name)

            label, speaker, index = file_name.split(".", 1)[0].split("_")
            index = int(index)

            if (index < 5 and what == "test") or (index >= 5 and what == "train"):
                self.file_names.append(file_name)

        np.random.RandomState(seed=seed).shuffle(self.file_names)
        N = len(self.file_names)
        self.file_names = self.file_names[: int(N * frac)]

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.file_names[self.last_idx :]) == 0:
            self.reroll(self.seed)
            raise StopIteration
        inp = []
        lab = []
        is_last = True
        last_idx = self.last_idx
        for i, f in enumerate(self.file_names[self.last_idx :]):
            fs, raw_wav = wav.read(self.root_dir + f)

            label, speaker, index = f.split(".", 1)[0].split("_")
            label = int(label)

            inp.append(raw_wav)
            lab.append(label)

            last_idx += 1
            if len(lab) >= self.batch_size:
                is_last = False
                break
        self.last_idx = last_idx
        return inp, lab

    def __len__(self):
        return len(self.file_names) // self.batch_size

    def reroll(self, seed=1000):
        np.random.RandomState(seed=seed).shuffle(self.file_names)
        self.last_idx = 0
