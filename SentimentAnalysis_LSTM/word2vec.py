import gensim
import torch
from preprocess import word2idx, vocab_size

wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
    "glove.6B.100d.w2vformat.txt", binary=False, encoding="utf-8"
)

## map golve pretrain weight to pytorch embedding pretrain weight
embed_size = 100
weight = torch.zeros(vocab_size + 1, embed_size)  # given 0 if the word is not in glove
for i in range(len(wvmodel.index_to_key)):
    try:
        index = word2idx[wvmodel.index_to_key[i]]  # transfer to our word2ind
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(wvmodel.index_to_key[i]))

if __name__ == "__main__":
    print(weight.shape)
