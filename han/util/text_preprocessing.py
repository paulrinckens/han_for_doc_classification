import re
import string

import spacy.lang.de.stop_words

PUNCTUATION = string.punctuation
STOP_WORDS = spacy.lang.de.stop_words.STOP_WORDS

nlp = spacy.load("de", disable=["tagger", "parser", "ner"])
nlp.add_pipe(nlp.create_pipe("sentencizer"))


def normalize(text: str):
    text = text.lower()
    text = text.strip()
    text = clean_str(text)
    doc = nlp(text)
    filtered_sentences = []
    for sentence in doc.sents:
        filtered_tokens = list()
        for i, w in enumerate(sentence):
            s = w.string.strip()
            if len(s) == 0 or s in PUNCTUATION and i < len(doc) - 1:
                continue
            if s not in STOP_WORDS:
                s = s.replace(',', '.')
                filtered_tokens.append(s)
        filtered_sentences.append(filtered_tokens)
    return filtered_sentences


def clean_str(string: str):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"<.*>", "", string)
    return string.strip()
