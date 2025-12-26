from typing import List, Optional, Sequence

from tokenization import text_segment as text_segment_module
from tokenization import token_merges as token_merges_module
from tokenization import vocabulary as vocabulary_module


class Tokenizer:
    def __init__(self):
        self._vocabulary: Optional[vocabulary_module.Vocabulary] = None
        self._token_merges: Optional[token_merges_module.TokenMerges] = None

    def train(
            self,
            text: str,
            vocab_size: int,
            special_token_texts: Sequence[str] = ["<|endoftext|>"]) -> None:
        preprocessed_chars = (
            vocabulary_module.extract_preprocessed_characters(text))
        if self._vocabulary is None:
            self._vocabulary = vocabulary_module.initialize_vocabulary(
                preprocessed_chars, special_token_texts)
        tokens = self._vocabulary.tokenize_preprocessed_characters(
            preprocessed_chars)
        if self._token_merges is None:
            self._token_merges = (
                token_merges_module.identify_byte_pair_encoding_merges(
                    tokens, len(self._vocabulary), vocab_size))
        self._token_merges.update_vocabulary(self._vocabulary)

    def encode(
            self, text: str, special_token_texts: Sequence[str]) -> List[int]:
        text_segments = text_segment_module.form_text_segments(
            text, special_token_texts)
        tokens: List[int] = []
        for text_segment in text_segments:
            tokens += text_segment.encode(self._vocabulary, self._token_merges)
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        token_texts = [
            self._vocabulary.get_text(token, raw=False) for token in  tokens]
        return ''.join(token_texts)
