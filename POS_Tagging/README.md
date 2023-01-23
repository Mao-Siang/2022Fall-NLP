# Part of Speech Recognition by spacy
This is my implementation of the Part of Speech recognition. For each sentence, the subject, verb, and object answers are provided. In this task, we need to check whether the provided answers are correct.

## Prerequisites
Install the spacy library by running the following:
```zsh
pip install spacy
```
Assume that numpy, pandas are installed. If not, install them with pip, too.

## Recognition Process
Analyze the input sentence by spacy first and find out the verb candidate by part of speech tagging first.
If the dependency tag of the word is `relcl` or the part of speech tag is `VERB` or `AUX`, it is considered as a verb candidate in this sentence.
Second, for each verb candidate, use the `subtree` method in spacy to find out the words that have relation with the verb. Third, find out the words in front of the verb as the candidates of subjects if their dependency tag includes `subj` and the words behind the verb as the candidates of objects if their dependency tag includes `obj`, `aux` or `cop`. Finally, check if the answer provided is included in the candidates.
