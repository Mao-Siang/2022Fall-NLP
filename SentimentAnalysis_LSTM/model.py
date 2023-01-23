"""
Author: Mao-Siang Chen
Student ID: 109550003
HW ID: HW2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from word2vec import weight, embed_size
from preprocess import vocab_size, word2idx


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        n_hiddens,
        n_layers,
        weight,
        bidirectional,
        labels,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.n_hiddens,
            num_layers=n_layers,
            bidirectional=self.bidirectional,
            dropout=0.1,
        )

        self.linear1 = nn.Linear(self.n_hiddens * 2, labels)

    def attention(self, lstm_output, hidden_output, input):
        # lstm_output = (len(sentence), batch_size, 2*n_hidden)
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in hidden_output], 1)

        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        """
        word = []
        for i in input[6].tolist():
            for w in word2idx.keys():
                if word2idx[w] == i:
                    word.append(w)
        print(word)
        print(f"focus on: {word[torch.argmax(weights[0])]}")
        """
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded.permute([1, 0, 2]))
        attn_output = self.attention(output, hidden, input)

        return self.linear1(attn_output.squeeze(0))


n_epochs = 20
n_hiddens = 24
n_layers = 1
bidirectional = True
labels = 7
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

net = RNN(
    vocab_size=(vocab_size + 1),
    embed_size=embed_size,
    n_hiddens=n_hiddens,
    n_layers=n_layers,
    bidirectional=bidirectional,
    weight=weight,
    labels=labels,
)

print(net)
