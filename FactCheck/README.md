# Fact Check by HuggingFace BERT
This is our implementation of the Fact Check Model.

## Prerequisites
Install the HuggingFace transformer library by running the following:
```zsh
pip install transformers
```
Assume that torch, numpy, pandas are installed. If not, install them with pip, too.

## Preprocess Data
The original data is get by web crawling from Twitter. We need to preprocess the data by extracting the useful information from the json file. Use the command:
```python
python3 compose.py
```
After preprocessing the data, there will be three files under your working directory.

## Training
Run the `train.py` command to train. The training process will automatically train with 3 epochs and a batch size 12. Last but not least, the learning rate is set to 3e-4. `f1 score` is used as the validation metrics.
The last model will be saved in `./model` directory.

## Testing
Run the `test.py` command to test. The testing process will automatically load the pretrained model and predict the results with the same settings as those in the training process. The results will be saved in output.csv under your working directory.
