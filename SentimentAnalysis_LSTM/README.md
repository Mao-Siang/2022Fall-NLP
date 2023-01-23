# Sentiment Analysis by LSTM model
This is my implementation of the Sentiment Analysis by LSTM model.

## Prerequisites
Install scikit-learn, nltk, gensim package by running:
```bash
pip install scikit-learn nltk gensim
```
Assume that torch, numpy, pandas are installed. If not, install them with pip, too.

## Preprocess Data
Since the data is not well distributed, multiply the number of data with respect to the labels. Then, use a self-defined tokenizer to extract the words and remove the punctuations. Finally, encode the words with glove encoding by gensim package. You can download the glove encoding [here](https://nlp.stanford.edu/data/glove.6B.zip).

## Training
You can train by ```python3 train.py```. The training process will automatically train with 20 epochs and a batch size 256. Cross entropy loss is used as the loss function. Last but not least, the learning rate is set to 2e-4.
The best and the last trained model's state dictionary will be saved under your working directory as `best_model.model` and `last_model.pt`.

The hyperparameters can be modified by yourself. `batch size` can be modified in `data.py`, `learning rate` can be modified in `train.py` and `epochs`, `hidden layer size` and other parameters about the model can be modified in `model.py`.
## Testing
You can run the testing process by ```python3 test.py```.
The results will be saved in `submission.csv` under your working directory.
