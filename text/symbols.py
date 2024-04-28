
""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_ipa_chars = ['ɟ', 'æ', 'k', 'iː', 'tʃ', 'd', 'dʒ', 'fʲ', 'd̪', 'ow', 'v', 'f', 'ŋ', 'ɱ', 'ɒ', 'ɲ', 'ɫ̩', 'cʰ', 'dʲ', 'ɝ', 'n', 'ɪ', 'ð', 'b', 'ɑ', 'pʰ', 'aj', 'tʷ', 'mʲ', 'ʉː', 'ç', 'ɛ', 'ɾ', 'ɟʷ', 'ɐ', 'm̩', 'h', 'm', 'kʷ', 'j', 'ɫ', 'ɔj', 'n̩', 'l', 'aw', 'z', 'p', 'ɡʷ', 'ɾ̃', 'ʉ', 'ɾʲ', 'kʰ', 'ʎ', 'ʔ', 'ʊ', 'θ', 'spn', 's', 'ej', 'ə', 'c', 'aː', 'ɒː', 'bʲ', 'ɹ', 'w', 'vʲ', 'i', 't̪', 'tʰ', 'ɡ', 'ʒ', 'ɚ', 'cʷ', 'pʲ', 'ʃ', 'pʷ', 't', 'tʲ', 'ɑː']
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_ipa = ["@" + s for s in _ipa_chars]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _ipa
    + _silences
)
