import logging
import pathlib
import pickle
from typing import List

import spacy
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.text import Tokenizer
from keras.utils import CustomObjectScope

from han.AttentionLayer import AttentionLayer
from han.util import embedding_utils
from han.util import text_preprocessing


class HAN:

    def __init__(self):
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None

        self.EMBED_DIM = 300
        self.MAX_WORD_NUM = 40
        self.MAX_SENTENCE_NUM = 9
        self.MAX_NUM_WORDS = 500000
        self.REG_PARAM = 1e-8

        self.logger = logging.getLogger()

        self.nlp = spacy.load("de", disable=["tagger", "parser", "ner"])
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

    def _build_model(self, embedding_matrix, n_classes):
        num_words = min(self.MAX_NUM_WORDS, len(self.tokenizer.word_index) + 1)

        # Word level attention model
        word_input = Input(shape=(self.MAX_WORD_NUM,), dtype="int32",
                           name="word_input")
        word_sequences = Embedding(num_words,
                                   self.EMBED_DIM,
                                   weights=[embedding_matrix],
                                   input_length=self.MAX_WORD_NUM,
                                   trainable=False,
                                   name="word_embedding")(word_input)
        word_gru = Bidirectional(GRU(50, return_sequences=True),
                                 name="word_gru")(word_sequences)
        word_dense = Dense(100, activation="relu", name="word_dense")(word_gru)
        word_att = AttentionLayer(self.EMBED_DIM, name="word_attention")(
            word_dense)
        word_encoder = Model(inputs=word_input, outputs=word_att)

        # Sentence level attention model
        sent_input = Input(shape=(self.MAX_SENTENCE_NUM, self.MAX_WORD_NUM),
                           dtype="int32", name="sent_input")
        sent_encoder = TimeDistributed(word_encoder, name="sent_linking")(
            sent_input)
        sent_gru = Bidirectional(GRU(50, return_sequences=True),
                                 name="sent_gru")(sent_encoder)
        sent_dense = Dense(100, activation="relu", name="sent_dense")(sent_gru)
        sent_att = AttentionLayer(self.EMBED_DIM, name="sent_attention")(
            sent_dense)
        sent_drop = Dropout(0.5, name="sent_dropout")(sent_att)
        preds = Dense(n_classes, activation="softmax", name="output")(sent_drop)

        model = Model(sent_input, preds)
        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=["acc"])

        self.logger.info(word_encoder.summary())
        self.logger.info(model.summary())

        return model, word_encoder

    def fit_tokenizer(self, texts: List[str]):
        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS, lower=True)
        tokenizer.fit_on_texts(texts)

        self.logger.info(f"Found {len(tokenizer.word_index)} unique tokens.")
        return tokenizer

    def build_and_train(self,
                        train_x,
                        train_y,
                        dev_x,
                        dev_y,
                        embedding_path: str,
                        model_dir: str,
                        model_filename: str,
                        tokenizer_filename: str,
                        epochs=10,
                        batch_size=32,
                        verbose: bool = False):

        # Fit tokenizer on train data
        self.tokenizer = self.fit_tokenizer(train_x)

        # Build embedding matrix from fitted tokenizer
        embedding_matrix = self._build_embedding_matrix(embedding_path)

        # Build model with embedding matrix
        self.model, self.word_attention_model = self._build_model(
            embedding_matrix, train_y.shape[1])

        # Encode train and dev data
        encoded_train_x = self.encode_texts(train_x)
        encoded_dev_x = self.encode_texts(dev_x)

        callbacks = []
        if model_dir:
            # save fitted toknizer
            tokenizer_path = str(pathlib.Path(model_dir, tokenizer_filename))
            with open(tokenizer_path, "wb") as handle:
                pickle.dump(self.tokenizer,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

            checkpoint_model_path = str(pathlib.Path(model_dir, model_filename))
            callbacks.append(ModelCheckpoint(checkpoint_model_path,
                                             verbose=verbose,
                                             monitor="val_loss",
                                             save_best_only=True,
                                             mode="auto"))

        self.model.fit(x=encoded_train_x,
                       y=train_y,
                       validation_data=(encoded_dev_x, dev_y),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks,
                       verbose=verbose)

    def load_model(self, model_dir: str, model_filename: str,
                   tokenizer_filename: str):
        # load sentence level attention model
        model_path = str(pathlib.Path(model_dir, model_filename))
        with CustomObjectScope({"AttentionLayer": AttentionLayer}):
            self.model = load_model(model_path)
            self.logger.info(f"Successfully loaded model from {model_path}")

        # retrieve word attention model
        self.word_attention_model = self.model.get_layer("sent_linking").layer

        # load tokenizer
        tokenizer_path = str(pathlib.Path(model_dir, tokenizer_filename))
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)
            self.logger.info(
                f"Successfully loaded tokenizer from {tokenizer_path}")

    def predict(self, texts: List[str]):
        encoded_text = self.encode_texts(texts)

        return self.model.predict(encoded_text)

    def predict_and_visualize_attention(self, text):
        encoded_text = self.encode_texts([text])[0]

        normalized_text = text_preprocessing.normalize(text)

        # word level attention model
        hidden_word_encoding_out = Model(
            inputs=self.word_attention_model.get_layer("word_input").output,
            outputs=self.word_attention_model.get_layer("word_dense").output)

        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
        word_context = self.word_attention_model.get_layer(
            "word_attention").get_weights()
        word_attentions = _get_attention_weights(hidden_word_encodings,
                                                 word_context)

        # sentence level attention model
        hidden_sent_encoding_out = Model(
            inputs=self.model.get_layer("sent_input").output,
            outputs=[self.model.get_layer("output").output,
                     self.model.get_layer("sent_dense").output])
        output_array = hidden_sent_encoding_out.predict(
            np.expand_dims(encoded_text, 0))
        sentence_attentions = _get_attention_weights(output_array[1],
                                                     self.model.get_layer(
                                                         "sent_attention").get_weights())

        prediction = output_array[0]

        attention_result = []
        out_sent_num = min(len(normalized_text), len(sentence_attentions))
        sum_s_att = sum(sentence_attentions[:out_sent_num])
        for i in range(out_sent_num):
            wa = word_attentions[i][:len(normalized_text[i])]
            wa = wa / wa.sum(axis=0, keepdims=1)
            attention_result.append([sentence_attentions[i] / sum_s_att,
                                     [[w, a] for w, a in
                                      zip(wa, normalized_text[i])]])

        return list(prediction[0]), attention_result

    def encode_texts(self, texts: List[str]):
        encoded_texts = np.zeros(
            (len(texts), self.MAX_SENTENCE_NUM, self.MAX_WORD_NUM))
        for i, text in enumerate(texts):
            normalized_text = text_preprocessing.normalize(text)
            for j, sent in enumerate(normalized_text[:min(len(normalized_text),
                                                          self.MAX_SENTENCE_NUM)]):
                for k, t in enumerate(sent):
                    if k < self.MAX_WORD_NUM:
                        try:
                            encoded_texts[i, j, k] = self.tokenizer.word_index[
                                t]
                        except:
                            pass
        return encoded_texts

    def _build_embedding_matrix(self, embedding_path):
        self.logger.info(f"Loading embeddings from file {embedding_path} ...")
        embeddings_index = embedding_utils.get_embedding_index(embedding_path)
        self.logger.info(f"Found {len(embeddings_index)} word embeddings.")

        # prepare embedding matrix that maps word indexs to their vectors
        self.logger.info("Building embedding matrix ...")
        num_words = min(self.MAX_NUM_WORDS, len(self.tokenizer.word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.EMBED_DIM))
        for word, i in self.tokenizer.word_index.items():
            if i >= self.MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.logger.info("Successfully built embedding matrix.")

        return embedding_matrix


def _get_attention_weights(sequence, weights):
    uit = np.dot(sequence, weights[0]) + weights[1]
    uit = np.tanh(uit)

    ait = np.dot(uit, weights[2])
    ait = np.squeeze(ait)
    ait = np.exp(ait)
    ait /= np.sum(ait)

    return ait
