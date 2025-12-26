import dataclasses
import re
from typing import List, Sequence

from tokenization import vocabulary as vocabulary_module
from tokenization import token_merges as token_merges_module


def _get_ordered_special_token_texts(
        special_token_texts: List[str]) -> List[str]:
    # Safeguard against edge case where one special token text is a subset of
    # another.
    return sorted(special_token_texts, key=len, reverse=True)

def _split_text_on_spaces_without_line_breaks(text: str) -> List[str]:
    space_segmented_texts: List[str] = []
    preceding_spaces = 0
    spaces_match_pattern = r'( +)|(\S+)'
    for match in re.finditer(spaces_match_pattern, text):
        # Accumulate preceding spaces if present.
        if match.group(1) is not None:
            preceding_spaces += len(match.group(1))
            continue
        word = match.group(2)
        # Add preceding spaces if present, reserving one for the word.
        if preceding_spaces:
            for _ in range(preceding_spaces - 1):
                space_segmented_texts.append(
                    vocabulary_module.SPACE_STRING_REPR)
            space_segmented_texts.append(
                f'{vocabulary_module.SPACE_STRING_REPR}{word}')
            preceding_spaces = 0
        else:
            space_segmented_texts.append(word)
    # Add trailing space characters.
    for _ in range(preceding_spaces):
        space_segmented_texts.append(vocabulary_module.SPACE_STRING_REPR)
    return space_segmented_texts

def _split_text_on_spaces(text: str) -> List[str]:
    space_segmented_texts: List[str] = []
    # Form groups outside of newline and carriage return characters.
    parts = re.split(r'(\r\n|\r|\n)', text)
    for part in parts:
        if part == "":
            continue
        if part == "\r\n":
            space_segmented_texts.append("\r")
            space_segmented_texts.append("\n")
            continue
        if part == "\r":
            space_segmented_texts.append("\r")
            continue
        if part == "\n":
            space_segmented_texts.append("\n")
            continue
        space_segmented_texts += (
            _split_text_on_spaces_without_line_breaks(part))
    return space_segmented_texts


@dataclasses.dataclass(frozen=True)
class TextSegment:
    text: str
    special: bool

    def _encode_special(
            self, vocabulary: vocabulary_module.Vocabulary) -> List[int]:
        if not self.special:
            raise ValueError(
                f'Cannot encode non-special text {self.text} as special.')
        return [vocabulary.get_token(self.text)]

    def _encode_standard(
            self,
            vocabulary: vocabulary_module.Vocabulary,
            token_merges: token_merges_module.TokenMerges) -> List[int]:
        if self.special:
            raise ValueError(
                f'Cannot encode special text {self.text} as standard text.')
        tokens: List[int] = []
        for text in _split_text_on_spaces(self.text):
            token = vocabulary.get_token(text)
            if token is not None:
                tokens.append(token)
            else:
                tokens += token_merges.tokenize_text_segment(text, vocabulary)
        return tokens

    def encode(
            self,
            vocabulary: vocabulary_module.Vocabulary,
            token_merges: token_merges_module.TokenMerges) -> List[int]:
        if self.special:
            return self._encode_special(vocabulary)
        return self._encode_standard(vocabulary, token_merges)


def form_text_segments(
        text: str, special_token_texts: Sequence[str]) -> List[TextSegment]:
    if not special_token_texts:
        return [TextSegment(text, special=False)]
    text_segments: List[TextSegment] = []
    sanitized_token_texts = [
        re.escape(token_text) for token_text in
        _get_ordered_special_token_texts(special_token_texts)]
    match_pattern = f'({'|'.join(sanitized_token_texts)})'
    text_idx = 0
    for match in re.finditer(match_pattern, text):
        standard_text = text[text_idx:match.start()]
        if standard_text:
            text_segments.append(TextSegment(standard_text, special=False))
            text_idx = match.start()
            continue
        special_text = match.group(0)
        text_segments.append(TextSegment(special_text, special=True))
        text_idx = match.start() + len(special_text)
    remaining_standard_text = text[text_idx:]
    if remaining_standard_text:
        text_segments.append(
            TextSegment(remaining_standard_text, special=False))
    return text_segments
