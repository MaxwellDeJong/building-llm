import collections
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

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


def _find_most_frequent_token_pair(tokens: Sequence[int]) -> (
        Optional[TokenPair]):
    pairs = collections.Counter(zip(tokens[:-1], tokens[1:]))
    if not pairs:
        return None
    token1, token2 = max(pairs.items(), key=lambda x: x[1])[0]
    return TokenPair(token1, token2)

def _replace_token_pair(
        tokens: Sequence[int],
        new_token: int,
        token_pair: TokenPair) -> List[int]:
    dq = collections.deque(tokens)
    replaced: List[int] = []
    while dq:
        current_token = dq.popleft()
        if dq and token_pair.is_match((current_token, dq[0])):
            replaced_token = new_token
            dq.popleft()
        else:
            replaced_token = current_token
        replaced.append(replaced_token)
    return replaced


@dataclasses.dataclass
class TokenMerges:
    merges: Dict[TokenPair, int]

    def update_vocabulary(
            self, vocabulary: vocabulary_module.Vocabulary) -> None:
        for token_pair, new_token in self.merges.items():
            merged_text = token_pair.form_merged_text(vocabulary)
            vocabulary.text_to_token[merged_text] = new_token
            vocabulary.token_to_text[new_token] = merged_text

    def __len__(self) -> int:
        return len(self.merges)


def identify_byte_pair_encoding_merges(
        tokens: Sequence[int],
        initial_vocab_size: int,
        vocab_size: int) -> None:
    merges: Dict[TokenPair, int] = {}
    for next_token in range(initial_vocab_size, vocab_size):
        token_pair = _find_most_frequent_token_pair(tokens)
        if token_pair is None:
            break
        tokens = _replace_token_pair(tokens, next_token, token_pair)
        merges[token_pair] = next_token
    return TokenMerges(merges)
