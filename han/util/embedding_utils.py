import logging

import numpy as np

logger = logging.getLogger()


def get_embedding_index(embedding_path: str):
    logger.info("Indexing word vectors.")

    embeddings_index = {}
    with open(embedding_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    logger.info(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index
