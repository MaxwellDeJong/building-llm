from typing import Optional, Sequence

from tokenization import vocabulary as vocabulary_module
from tokenization import token_pairs as token_pairs_module


class Tokenizer:
    def __init__(self):
        self._vocabulary: Optional[vocabulary_module.Vocabulary] = None
        self._token_merges: Optional[token_pairs_module.TokenMerges] = None

    def train(
            self,
            text: str,
            vocab_size: int,
            special_tokens: Sequence[str] = ["<|endoftext|>"]) -> None:
        preprocessed_chars = (
            vocabulary_module.extract_preprocessed_characters(text))
        if self._vocabulary is None:
            self._vocabulary = vocabulary_module.initialize_vocabulary(
                preprocessed_chars, special_tokens)
        tokens = self._vocabulary.tokenize_characters(preprocessed_chars)
        if self._token_merges is None:
            self._token_merges = (
                token_pairs_module.identify_byte_pair_encoding_merges(
                    tokens, len(self._vocabulary), vocab_size))
        self._token_merges.update_vocabulary(self._vocabulary)
