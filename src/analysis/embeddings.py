from gensim.models import KeyedVectors, FastText as GensimFastText
from sentence_transformers import SentenceTransformer
import numpy as np

# BERT
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# Word2Vec (GoogleNews-vectors)
def load_word2vec(path='GoogleNews-vectors-negative300.bin'):
    return KeyedVectors.load_word2vec_format(path, binary=True)


# GloVe
def load_glove(path='glove.6B.300d.txt'):
    return KeyedVectors.load_word2vec_format(path, binary=False, no_header=True)


# FastText
def load_fasttext(path='cc.en.300.bin'):
    return GensimFastText.load_fasttext_format(path)


def get_average_embedding(tokens, model, model_type='gensim'):
    vectors = []
    for token in tokens:
        try:
            if model_type == 'sentence':
                vec = bert_model.encode(token)
            else:
                vec = model[token]
            vectors.append(vec)
        except:
            continue
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)
