import logging

import nltk
from nltk.corpus import words
from nltk.data import find

import tangerine.config as cfg

log = logging.getLogger("tangerine.nltk")


def init_nltk():
    # Check for the words corpus in the search path
    try:
        find("corpora/words")
    except LookupError:
        log.info(f"Downloading NLTK words corpus to {cfg.NLTK_DATA_DIR}...")
        nltk.download("words", quiet=True, download_dir=cfg.NLTK_DATA_DIR)


def get_words():
    init_nltk()
    return set(words.words())
