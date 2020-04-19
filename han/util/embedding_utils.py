import numpy as np


def get_embedding_index(embedding_path: str):
    embeddings_index = {}
    with open(embedding_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embeddings_index
