"""Common functionality for analyzers."""

from __future__ import annotations

import abc
import functools
import unicodedata

import annif

logger = annif.logger

_KEY_TOKEN_MIN_LENGTH = "token_min_length"
_NLTK_TOKENIZER_DATA = "punkt_tab"


class Analyzer(metaclass=abc.ABCMeta):
    """Base class for language-specific analyzers. Either tokenize_words or
    _normalize_word must be overridden in subclasses. Other methods may be
    overridden when necessary."""

    name = None
    token_min_length = 3  # default value, can be overridden in instances

    @staticmethod
    def is_available() -> bool:
        """Return True if the analyzer is available for use, False if not."""
        return True  # can be overridden in implementations if necessary

    def __init__(self, **kwargs) -> None:
        if _KEY_TOKEN_MIN_LENGTH in kwargs:
            self.token_min_length = int(kwargs[_KEY_TOKEN_MIN_LENGTH])

        import nltk.data

        try:
            nltk.data.find("tokenizers/" + _NLTK_TOKENIZER_DATA)
        except LookupError as err:
            logger.debug(str(err))
            if _NLTK_TOKENIZER_DATA in str(err):
                logger.warning(
                    f'NLTK datapackage "{_NLTK_TOKENIZER_DATA}" not found, '
                    "downloading it now."
                )
                nltk.download(_NLTK_TOKENIZER_DATA)
            else:
                raise

    def tokenize_sentences(self, text: str) -> list[str]:
        """Tokenize a piece of text (e.g. a document) into sentences."""
        import nltk.tokenize

        return nltk.tokenize.sent_tokenize(text)

    @functools.lru_cache(maxsize=50000)
    def is_valid_token(self, word: str) -> bool:
        """Return True if the word is an acceptable token."""
        if len(word) < self.token_min_length:
            return False
        for char in word:
            category = unicodedata.category(char)
            if category[0] == "L":  # letter
                return True
        return False

    def tokenize_words(self, text: str, filter: bool = True) -> list[str]:
        """Tokenize a piece of text (e.g. a sentence) into words. If
        filter=True (default), only return valid tokens (e.g. not
        punctuation, numbers or very short words)"""

        import nltk.tokenize

        return [
            self._normalize_word(word)
            for word in nltk.tokenize.word_tokenize(text)
            if (not filter or self.is_valid_token(word))
        ]

    def _normalize_word(self, word):
        """Normalize (stem or lemmatize) a word form into a normal form."""
        pass  # pragma: no cover
