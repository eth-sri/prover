import numpy as np
import scipy.io.wavfile as wav
import os, pickle


class DataStream:
    def __init__(self, what, batch=100, seed=1000, frac=1, root_dir=None):
        self.root_dir = "data/gsc/" if root_dir is None else root_dir
        self.what = what
        self.last_idx = 0
        self.file_names = []
        self.batch_size = batch
        self.seed = seed

        testing_list = []
        with open(self.root_dir + "testing_list.txt", "r") as test_files:
            for r in test_files:
                r = r.replace("\n", "")
                testing_list.append(r)
        validation_list = []
        with open(self.root_dir + "validation_list.txt", "r") as valid_files:
            for r in valid_files:
                r = r.replace("\n", "")
                validation_list.append(r)

        self.label_dict = {}
        label_idx = 0
        for subdir, _, files in sorted(os.walk(self.root_dir)):
            if "_background_noise_" in subdir or subdir[-1] == "/":
                continue
            label_str = subdir.split("/")[-1]
            self.label_dict[label_str] = label_idx
            label_idx += 1
            for f in sorted(files):
                in_test = label_str + "/" + f in testing_list
                in_valid = label_str + "/" + f in validation_list
                if (
                    (what == "test" and in_test)
                    or (what == "valid" and in_valid)
                    or (what == "train" and not in_test and not in_valid)
                ):
                    self.file_names.append(label_str + "/" + f)

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
            inp.append(raw_wav[::2])
            lab.append(self.label_dict[f.split("/")[0]])
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
