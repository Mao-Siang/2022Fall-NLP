import nltk
from nltk.corpus import stopwords
import pandas as pd
from itertools import chain
import numpy as np

# nltk.download("stopwords")

emo_to_idx = {
    "neutral": 0,
    "anger": 1,
    "joy": 2,
    "surprise": 3,
    "sadness": 4,
    "disgust": 5,
    "fear": 6,
}


def readCSV(path, isTest=False, isValid=False):
    df = pd.read_csv(path)
    if isTest:
        return df["Utterance"].values

    result = df[["Utterance", "Emotion"]].values

    for i in range(len(result)):
        result[i][1] = emo_to_idx[result[i][1]]

        if isValid != True:
            if result[i][1] == 1:
                result = np.append(result, [result[i] for _ in range(3)], axis=0)
            elif result[i][1] == 2:
                result = np.append(result, [result[i] for _ in range(2)], axis=0)
            elif result[i][1] == 3:
                result = np.append(result, [result[i] for _ in range(4)], axis=0)
            elif result[i][1] == 4:
                result = np.append(result, [result[i] for _ in range(7)], axis=0)
            elif result[i][1] == 5:
                result = np.append(result, [result[i] for _ in range(10)], axis=0)
            elif result[i][1] == 6:
                result = np.append(result, [result[i] for _ in range(10)], axis=0)
    return result


def tokenizer(text):
    token = nltk.word_tokenize(text.lower())
    proc_token = []
    for word in token:
        if word in [",", ".", "'"] or word in stopwords.words("English"):
            continue
        else:
            proc_token.append(word)
    # print(token)
    return proc_token


def encode_sentence(data):
    result = []
    for text in data:
        vector = []
        for word in text:
            if word in word2idx:
                vector.append(word2idx[word])
            else:
                vector.append(0)
        result.append(vector)
    return result


def pad_sentence(data, length=10):
    result = []
    for text in data:
        if len(text) >= length:
            result.append(text[:length])
        else:
            pad_text = text
            while len(pad_text) < length:
                pad_text.append(0)
            result.append(pad_text)
    return result


"""
read data
"""
train_data = readCSV("./data/train_HW2dataset.csv")
dev_data = readCSV("./data/dev_HW2dataset.csv", isValid=True)
test_data = readCSV("./data/test_HW2dataset.csv", isTest=True)

"""
tokenize
"""
train_tokenized = []
dev_tokenized = []
test_tokenized = []

for sent, label in train_data:
    train_tokenized.append(tokenizer(sent))


for sent, label in dev_data:
    dev_tokenized.append(tokenizer(sent))

for sent in test_data:
    test_tokenized.append(tokenizer(sent))


vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)

"""
word encoding 
"""
word2idx = {word: i + 1 for i, word in enumerate(vocab)}
word2idx["<unk>"] = 0


if __name__ == "__main__":
    a = readCSV("./data/dev_HW2dataset.csv")
