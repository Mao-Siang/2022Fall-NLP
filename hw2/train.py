import torch.nn as nn
import torch
from torch import optim
from model import net, device, n_epochs
from data import train_iter, val_iter, test_iter
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import time
import os
import pandas as pd

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-4)

model_save_path = "./"
torch.manual_seed(27)


def train(net, num_epochs, loss_function, optimizer, train_iter, val_iter):
    best_valid = 100
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        net.train()
        for feature, label in train_iter:
            n += 1

            optimizer.zero_grad()
            feature = Variable(feature.to(device))
            label = Variable(label.to(device))

            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(
                torch.argmax(score.cpu().data, dim=1), label.cpu()
            )
            train_loss += loss

        with torch.no_grad():
            net.eval()
            for val_feature, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(
                    torch.argmax(val_score.cpu().data, dim=1), val_label.cpu()
                )
                val_losses += val_loss

        runtime = time.time() - start
        print(
            "epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f"
            % (
                epoch,
                train_loss.data / n,
                train_acc / n,
                val_losses.data / m,
                val_acc / m,
                runtime,
            )
        )
        if val_losses.data / m < best_valid:
            best_valid = val_losses.data / m
            torch.save(
                net.state_dict(), os.path.join(model_save_path, "best_model.model")
            )

    # save final model
    state = {
        "epoch": epoch,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_save_path, "last_model.pt"))


def predict(net, test_iter):

    pred_list = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        net.eval()
        for batch, _ in test_iter:
            output = net(batch.to(device))
            pred_list.extend(torch.argmax(softmax(output), dim=1).cpu().numpy())

    return pred_list


print("start to train...")
train(net, n_epochs, loss_function, optimizer, train_iter, val_iter)

print("start to predict test set...")
pred_list = predict(net, test_iter)

df = pd.DataFrame(data={"emotion": pred_list})
df.to_csv("./submission.csv", index_label="index")
print("Done")
