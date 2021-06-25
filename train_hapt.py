from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import hapt_loader
from models_mnist import MnistModel
from utils import get_default_device


def train_model(epochs=100, lr=0.001, batch_size=30):
    print("Training data loading")
    train_loader = hapt_loader.DataStream("train", batch=batch_size, max_len=300)
    print("Test data loading")
    valid_loader = hapt_loader.DataStream("test", batch=batch_size, max_len=300)

    model = MnistModel(75, 256, 4, 6).to(device=get_default_device())
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

        torch.save(model.state_dict(), "saved/hapt_4L_256H.pt")


if __name__ == "__main__":
    train_model(epochs=30, lr=0.001, batch_size=32)
