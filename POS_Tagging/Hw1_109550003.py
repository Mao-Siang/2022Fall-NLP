# Author: Mao-Siang Chen
# Student ID: 109550003
# HW ID: HW1

import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("./dataset.csv", header=None)

result = []

for i in range(len(df)):
    sentence = str(df.loc[i][1])
    doc = nlp(sentence)

    verb = []

    for token in doc:
        if token.pos_ == "VERB" or token.dep_ == "relcl" or token.pos_ == "AUX":
            verb.append(token)

    candidate = []
    for v in verb:
        subj = []
        obj = []

        for word in v.subtree:
            if ("subj" in word.dep_) and sentence.find(word.text) < sentence.find(
                v.text
            ):
                subj.append(word.text)
                if word.text == "who":
                    dobj = ""
                    for voc in v.subtree:
                        if ("PROPN" in voc.pos_) and sentence.find(
                            voc.text
                        ) < sentence.find(word.text):
                            dobj = voc.text
                    subj.append(dobj)
            elif (
                "obj" in word.dep_ or "aux" in word.dep_ or "cop" in word.dep_
            ) and sentence.find(word.text) > sentence.find(v.text):
                if len(obj) < 2:
                    obj.append(word.text)

        for s in subj:
            for o in obj:
                candidate.append([s, v.text, o])

    ans = 0
    for pair in candidate:
        if (
            pair[0] in str(df.loc[i][2])
            and pair[1] in str(df.loc[i][3])
            and pair[2] in str(df.loc[i][4])
        ):
            ans = 1
            break
    result.append(ans)

output = pd.DataFrame(data={"T/F": result})
output.to_csv("./submission.csv", index_label="index")
