import torch
from preprocess import *


train_features = torch.tensor(pad_sentence(encode_sentence(train_tokenized)))
train_labels = torch.tensor([data[1] for data in train_data])
val_features = torch.tensor(pad_sentence(encode_sentence(dev_tokenized)))
val_labels = torch.tensor([data[1] for data in dev_data])
test_features = torch.tensor(pad_sentence(encode_sentence(test_tokenized)))
test_labels = torch.tensor([0 for _ in range(len(test_features))])

batch_size = 256
train_set = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = torch.utils.data.TensorDataset(val_features, val_labels)
val_iter = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

test_set = torch.utils.data.TensorDataset(test_features, test_labels)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
