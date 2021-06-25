import torch, math
import numpy as np


# General utility functions
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def negative_only(w):
    return -torch.relu(-w)


def positive_only(w):
    return torch.relu(w)


# Below methods are for the speech preprocessing stages. Most of the things are
# for algebraic functions and definitions.
def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def get_filterbanks(nfilt=26, nfft=512, samplerate=16000, lowfreq=0, highfreq=8000):
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


class M:
    def __init__(
        self,
        frame_size=256,
        frame_step=200,
        n_filt=10,
        samprate=8000,
        dev=torch.device("cpu"),
    ):
        N = frame_size
        s = frame_step
        n = N // 2 + 1

        self.preemph = torch.Tensor(np.eye(N) - np.eye(N, k=1) * 0.97).to(device=dev)
        self.hamming = torch.hamming_window(N).to(device=dev)
        self.W = np.array(
            [
                [
                    [math.cos(i * k * -2.0 * math.pi / N) for k in range(n)]
                    for i in range(N)
                ],
                [
                    [math.sin(i * k * -2.0 * math.pi / N) for k in range(n)]
                    for i in range(N)
                ],
            ],
            dtype=np.float32,
        )  # W[0, ]: real, W[1, ]: imaginary
        self.W_comb = torch.Tensor(
            np.concatenate([self.W[0], self.W[1]], axis=1) / np.sqrt(N)
        ).to(device=dev)

        self.real_add_img = torch.Tensor(np.concatenate([np.eye(n), np.eye(n)], 0)).to(
            device=dev
        )
        self.fb = torch.Tensor(
            get_filterbanks(n_filt, N, samprate, 0, samprate // 2).T.astype(np.float32)
        ).to(device=dev)
