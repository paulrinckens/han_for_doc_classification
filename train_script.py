"""
Simple script to train a new model and tokenizer on the 10kGNAD dataset.
"""
from han.HAN import HAN
from han.util import TenKGNAD

han = HAN()

train_x, train_y, dev_x, dev_y, test_x, test_y, class_names = TenKGNAD.load(
    "./data/articles.csv")

han.build_and_train(train_x=train_x,
                    train_y=train_y,
                    dev_x=dev_x,
                    dev_y=dev_y,
                    embedding_path="./embeddings/glove_german/vectors.txt",
                    model_dir="./models",
                    model_filename="han-10kGNAD.h5",
                    tokenizer_filename="tokenizer.pickle",
                    epochs=10,
                    batch_size=32,
                    verbose=True)
