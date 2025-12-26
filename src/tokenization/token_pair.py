import dataclasses
from typing import Tuple

from tokenization import vocabulary as vocabulary_module


@dataclasses.dataclass(frozen=True)
class TokenPair:
    token1: int
    token2: int

    def is_match(self, pair_tuple: Tuple[int, int]) -> bool:
        if len(pair_tuple) != 2:
            raise ValueError(
                f'Equality requires a tuple of length 2 rather than '
                f'{len(pair_tuple)}.')
        return (self.token1 == pair_tuple[0]) and (
                self.token2 == pair_tuple[1])

    def form_merged_text(
            self, vocabulary: vocabulary_module.Vocabulary) -> str:
        return (vocabulary.token_to_text[self.token1] +
                vocabulary.token_to_text[self.token2])

