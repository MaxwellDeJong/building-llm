import collections
import dataclasses
from typing import Dict, List, Optional, Sequence

from tokenization import token_pair as token_pair_module
from tokenization import vocabulary as vocabulary_module


def _find_most_frequent_token_pair(tokens: Sequence[int]) -> (
        Optional[token_pair_module.TokenPair]):
    pairs = collections.Counter(zip(tokens[:-1], tokens[1:]))
    if not pairs:
        return None
    token1, token2 = max(pairs.items(), key=lambda x: x[1])[0]
    return token_pair_module.TokenPair(token1, token2)

def _replace_token_pair(
        tokens: Sequence[int],
        new_token: int,
        token_pair: token_pair_module.TokenPair) -> List[int]:
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
    merges: Dict[token_pair_module.TokenPair, int]

    def __contains__(self, token_pair: token_pair_module.TokenPair) -> bool:
        return token_pair in self.merges

    def __len__(self) -> int:
        return len(self.merges)

    def update_vocabulary(
            self, vocabulary: vocabulary_module.Vocabulary) -> None:
        for token_pair, new_token in self.merges.items():
            merged_text = token_pair.form_merged_text(vocabulary)
            vocabulary.text_to_token[merged_text] = new_token
            vocabulary.token_to_text[new_token] = merged_text

    def tokenize_text_segment(
            self, 
            text_segment: str,
            vocabulary: vocabulary_module.Vocabulary) -> List[int]:
        tokens = [vocabulary.text_to_token.get(char) for char in text_segment]
        additional_merges_possible = True
        while additional_merges_possible and len(tokens) > 1:
            additional_merges_possible = False
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens) - 1:
                token_pair = token_pair_module.TokenPair(
                    tokens[i], tokens[i+ 1])
                # If the token pair has been marked as a merger, add the merged
                # token and increment the pointer past these tokens.
                if token_pair in self.merges:
                    new_tokens.append(self.merges[token_pair])
                    i += 2
                    additional_merges_possible = True
                # If the token pair has not been marked as a merger, simply add
                # the current token.
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            # Add any remaining tokens.
            if i < len(tokens):
                new_tokens.append(tokens[i])
            tokens = new_tokens
        return tokens


def identify_byte_pair_encoding_merges(
        tokens: Sequence[int],
        initial_vocab_size: int,
        vocab_size: int) -> None:
    merges: Dict[token_pair_module.TokenPair, int] = {}
    for next_token in range(initial_vocab_size, vocab_size):
        token_pair = _find_most_frequent_token_pair(tokens)
        if token_pair is None:
            break
        tokens = _replace_token_pair(tokens, next_token, token_pair)
        merges[token_pair] = next_token
    return TokenMerges(merges)
