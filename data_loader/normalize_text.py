# Normalize and clean Japanese characters
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

CONVERSION_TABLE = {
    #replace
    '×': 'x',
    '∼': '~', '〜': '~',
    '〔': '(', '（': '(', '）': ')',
    '＆': '&',
    '‐': '-', '–': '-', '―': '-', '−': '-', '─': '-', 'ー': '-', '—': '-',
    '【': '[', '】': ']',
    '`': "'",
    # '、': ',',
    '·': '・', '•': '・',
    '“': '"', '”': '"',
    # '。': '.',


    ###Remove, replace by unknown character
    '§': '♥',
    '̉': '♥',
    '̣': '♥',
    '○': '♥',
    '〇': '♥',
    '©': '♥',
    '®': '♥',
    'Á': '♥',
    'é': '♥',
    'Đ': '♥',
    '̣̉': '♥',
    '̻': '♥',
    '※': '♥',
    '→': '♥',
    '⇔': '♥',
    '≪': '♥',
    '≫': '♥',
    '■': '♥',
    '▣': '♥',
    '▲': '♥',
    '△': '♥',
    '▼': '♥',
    '◆': '♥',
    '◎': '♥',
    '●': '♥',
    '★': '♥',
    '☆': '♥',
    '➡': '♥',
    '《': '♥',
    '》': '♥',
    '※': '♥',


    # Katakana Lower 2 Upper
    'ㇱ': 'シ',
    'ㇰ': 'ク',
    'ヮ': 'ワ',
    '厶': 'ム',
    'ㇼ': 'リ',
    'ㇻ': 'ラ',
    'ㇽ': 'ル',
    'ㇷ': 'フ',
    'ヾ': 'ミ',
    'ㇳ': 'ト',
    'ㇴ': 'ヌ',
    'ㇵ': 'ハ',
    'ㇶ': 'ヒ',
    'ㇹ': 'ホ',
    'ㇺ': 'ム',
    'ㇾ': 'レ',
    'ㇿ': 'ロ',

    # Hiragana Lower 2 Upper
    'ゅ': 'ゆ',
    'ょ': 'よ',
    'ゃ': 'や',
    'ぃ': 'い',
    'ぁ': 'あ',
    'ぇ': 'え',
    'ぉ': 'お',
    'っ': 'つ',
    'ぅ': 'う',
    'ㇲ': 'ス',
    'ゕ': 'か',
    'ゖ': 'け',
    'ゎ': 'わ',
    'ㇸ': 'ヘ',
    'へ': 'ヘ',
    'ぺ': 'ペ',
    'べ': 'ベ',
    'ぱ': 'ぱ',
    'ぽ': 'ぽ',
}

import unicodedata

def hira_to_kata(hira_char):
    """Convert hiragana character to katakana character
    # Arguments
        hira_char [str]: the hiragana character
    # Returns
        [str]: the corresponding katakana character
    """
    if ord(hira_char) < 12354 or ord(hira_char) > 12447:
        raise ValueError('`hira_char` should be hiragana character')

    return chr(ord(hira_char) + 96)


def kata_to_hira(kata_char):
    """Convert katakana character to hiragana character
    # Arguments
        kata_char [str]: the katakana character
    # Returns
        [str]: the corresponding hiragana character
    """
    if ord(kata_char) < 12450 or ord(kata_char) > 12542:
        raise ValueError('`kata_char` should be a katakana character')

    return chr(ord(kata_char) - 96)


def load_conversion_table(update=None):
    """Return a conversion table.
    # Arguments
        update [dict]: extra items to update
    # Returns
        [dict]: conversion table
    """
    table = {}
    table.update(CONVERSION_TABLE)
    if update is not None:
        table.update(update)

    return table


def normalize_char(char, conversion_table=None):
    """Normalize Unicode character
    # Arguments
        char [str]: the character we wish to normalize
        conversion_table [dict]: the dictionary that contains keys that are
            not supported by unicodedata
    # Returns
        [str]: the normalized character
    """
    if conversion_table is None:
        conversion_table = CONVERSION_TABLE

    char = unicodedata.normalize('NFKC', char)
    return conversion_table.get(char, char)


def normalize_text(text, conversion_table=None):
    """Normalize the Unicode text
    # Argument
        text [str]: the text to normalize
        conversion_table [dict]: the dictionary that contains keys that are
            not supported by unicodedata
    # Returns
        [str]: the normalized text
    """
    if conversion_table is None:
        conversion_table = CONVERSION_TABLE

    chars = []
    for each_char in text:
        chars.append(normalize_char(each_char, conversion_table))

    return ''.join(chars)


if __name__ == "__main__":
    """ Unit test """
    text = "イリヨウホウジントクシユウカイナガサキキタトクシユウカイビヨウインㇸㇸㇸ"
    norm_text = normalize_text(text)
    print(text, '\n', norm_text)