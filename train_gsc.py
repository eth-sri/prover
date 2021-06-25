from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import gsc_loader
from models_speech import SpeechClassifier
from utils import get_default_device


def train_model(epochs=100, lr=0.001, batch_size=30):
    print("Training data loading")
    train_loader = gsc_loader.DataStream("train", batch_size)
    print("Test data loading")
    valid_loader = gsc_loader.DataStream("test", batch_size)
    model = SpeechClassifier(1024, 1024, 10, 50, 50, 1, 35, batch_size, 8000).to(
        device=get_default_device()
    )
    opt_fn = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        epoch_loss = 0.0

        totlen = 0
        for inp, lab in tqdm(train_loader):
            if len(inp) < batch_size:
                continue
            maxlen = max([np.shape(x)[0] for x in inp])
            batch_input = torch.Tensor(
                [np.concatenate([[0] * (maxlen - np.shape(x)[0]), x], 0) for x in inp]
            ).to(device=get_default_device())
            batch_label = torch.Tensor(lab).to(
                dtype=torch.long, device=get_default_device()
            )

            pred = model(batch_input)
            loss = F.cross_entropy(pred, batch_label)
            epoch_loss += loss.item() * batch_label.size()[0]
            totlen += batch_label.size()[0]

            loss.backward()
            opt_fn.step()
            opt_fn.zero_grad()

        print(f"\tAvg. training loss: {epoch_loss/totlen}")

        if (epoch + 1) % 3 == 0:
            correct_guess = 0.0
            val_loss = 0.0
            with torch.no_grad():
                # for i in tqdm(range(0, len(test_input), batch_size)):
                totlen = 0
                for inp, lab in tqdm(valid_loader):
                    if len(inp) < batch_size:
                        continue
                    maxlen = max([np.shape(x)[0] for x in inp])
                    batch_input = torch.Tensor(
                        [
                            np.concatenate([[0] * (maxlen - np.shape(x)[0]), x], 0)
                            for x in inp
                        ]
                    ).to(device=get_default_device())
                    batch_label = torch.Tensor(lab).to(
                        dtype=torch.long, device=get_default_device()
                    )

                    pred = model(batch_input)
                    loss = F.cross_entropy(pred, batch_label)

                    correct_guess += torch.sum(pred.argmax(1) == batch_label).item()
                    val_loss += loss.item() * batch_label.size()[0]
                    totlen += batch_label.size()[0]

            print(f"----validation loss: {val_loss/totlen}")
            print(f"----accuracy: {correct_guess/totlen*100.}%")

        torch.save(model.state_dict(), "saved/gsc.pt")


if __name__ == "__main__":
    train_model(epochs=30, lr=0.001, batch_size=100)
