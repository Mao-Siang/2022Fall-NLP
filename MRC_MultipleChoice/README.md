# Machine Reading Comprehension with multiple choice by HuggingFace BERT
This is our implementation of the MRC multiple choice model.

## Prerequisites
Install the HuggingFace transformer library by running the following:
```zsh
pip install transformers
```
Assume that torch, numpy, pandas are installed. If not, install them with pip, too.

## Preprocess Data
Since the number of choices is not the same, we need to pad empty strings. Then, sort the articles based on their length to reduce the number of unnecessary paddings.

## Training
Run the python notebook to train. The training process will automatically train with 4 epochs and a batch size 2. Last but not least, the learning rate is set to 1e-5. `accuracy` is used as the validation metrics.
The trained model will be saved in `./MRC_MODEL` directory after finishing each epoch.

If your GPU is strong enough, you can try larger batch sizes.
## Testing
Testing section will start after the training section has finished in the python notebook. 
The testing process will load the pretrained model and predict the results with the same settings as those in the training process. The results will be saved in `submission.csv` under your working directory.
