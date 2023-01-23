import torch
import torch.nn as nn
from model import RNN, device, net
from data import test_iter
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net
model.to(device)
model.load_state_dict(torch.load("best_model.model"))


def predict(net, test_iter):

    pred_list = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        net.eval()
        for batch, _ in test_iter:
            output = net(batch.to(device))
            pred_list.extend(torch.argmax(softmax(output), dim=1).cpu().numpy())

    return pred_list


print("start to predict test set...")
pred_list = predict(model, test_iter)

df = pd.DataFrame(data={"emotion": pred_list})
df.to_csv("./submission_1.csv", index_label="index")
print("Done")
