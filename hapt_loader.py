import numpy as np
import os


class DataStream:
    def __init__(self, what, batch=100, seed=1000, samp_over=5, max_len=50):
        self.root_dir = "data/HAPT/RawData/"
        self.what = what
        self.last_idx = 0
        self.batch_size = batch
        self.seed = seed

        self.ang_speed = []
        self.tag_idx = 0

        with open(os.path.join(self.root_dir, "labels.txt"), "r") as fl:
            recent_pair = (0, 0)
            recent_data = []
            for row in fl:
                row_sep = [int(x) for x in row.split(" ")]
                exp_id, usr_id, act_id, st, ed = row_sep
                act_id -= 1

                if act_id >= 6:  # excluding transition
                    continue

                if recent_pair != (exp_id, usr_id):
                    recent_data = []
                    recent_pair = (exp_id, usr_id)
                    with open(
                        os.path.join(
                            self.root_dir, f"gyro_exp{exp_id:02d}_user{usr_id:02d}.txt"
                        ),
                        "r",
                    ) as new_fl:
                        for new_row in new_fl:
                            recent_data.append([float(x) for x in new_row.split(" ")])

                seq_chunk = recent_data[st:ed]
                seq = []
                for i in range(min(max_len, len(seq_chunk))):
                    seq.extend(seq_chunk[i])
                self.ang_speed.append((seq, act_id, self.tag_idx))

                for j in range(max_len, len(seq_chunk)):
                    seq = seq[3:] + seq_chunk[j]
                    if j % samp_over == 0:
                        self.ang_speed.append((seq, act_id, self.tag_idx))

                self.tag_idx += 1

        self.reroll(seed=seed)
        self.tags = np.arange(self.tag_idx)
        np.random.RandomState(seed=seed).shuffle(self.tags)
        N = self.tag_idx
        self.split = {
            "train": self.tags[: int(N * 0.7)],
            "test": self.tags[int(N * 0.7) :],
        }

        print("data prepared")
        print("train split:", len(self.split["train"]), "chunks")
        print("test split:", len(self.split["test"]), "chunks")

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.ang_speed[self.last_idx :]) == 0:
            self.reroll(self.seed)
            raise StopIteration
        inp = []
        lab = []
        is_last = True
        last_idx = self.last_idx
        for seq, label, tag in self.ang_speed[self.last_idx :]:
            last_idx += 1
            if tag not in self.split[self.what]:
                continue

            inp.append(seq)
            lab.append(label)
            if len(lab) >= self.batch_size:
                is_last = False
                break

        self.last_idx = last_idx
        return inp, lab

    def __len__(self):
        return len(self.ang_speed) // self.batch_size

    def reroll(self, seed=1000):
        np.random.RandomState(seed=seed).shuffle(self.ang_speed)
        self.last_idx = 0
