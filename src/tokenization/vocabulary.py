import dataclasses
from typing import Dict, List, Sequence

SPACE_STRING_REPR = 'Ġ'


def extract_preprocessed_characters(text: str) -> List[str]:
    preprocessed_chars: List[str] = []
    for i, char in enumerate(text):
        if char == " " and i != 0:
            preprocessed_chars.append(SPACE_STRING_REPR)
        if char != " ":
            preprocessed_chars.append(char)
    return preprocessed_chars

def _identify_unique_characters(
        preprocessed_chars: List[str],
        special_tokens: Sequence[str]) -> List[str]:
    # Initialize vocab with with the first 256 ASCII characters.
    initial_chars = list(map(chr, range(256)))
    print(f'Initial chars: {list(initial_chars)}')
    # Add space character and special tokens for consideration.
    candidate_additional_chars = (
        preprocessed_chars + [SPACE_STRING_REPR] + list(special_tokens))
    print(f'Candidate initial characters set: {sorted(set(candidate_additional_chars))}')
    additional_chars = [
        char for char in sorted(set(candidate_additional_chars))
        if char not in initial_chars]
    return initial_chars + additional_chars


@dataclasses.dataclass
class Vocabulary:
    token_to_text: Dict[int, str]
    text_to_token: Dict[str, int]

    def __len__(self) -> int:
        if len(self.token_to_text) != len(self.text_to_token):
            raise ValueError(
                f'Invalid dictionaries of unequal length. Token to text has '
                f'length {len(self.token_to_text)} while text to token has '
                f'length {len(self.text_to_token)}.')
        return len(self.token_to_text)

    def tokenize_characters(
            self, preprocessed_characters: Sequence[str]) -> List[int]:
        return [self.text_to_token[char] for char in preprocessed_characters]


def initialize_vocabulary(
        preprocessed_chars: Sequence[str],
        special_tokens: Sequence[str]) -> Vocabulary:
    token_to_text: Dict[int, str] = {}
    text_to_token: Dict[str, int] = {}
    unique_chars = _identify_unique_characters(
        preprocessed_chars, special_tokens)
    print(f'Received a total of {len(unique_chars)} unique chars.')
    for i, char in enumerate(unique_chars):
        token_to_text[i] = char
        text_to_token[char] = i
    return Vocabulary(token_to_text, text_to_token)
 