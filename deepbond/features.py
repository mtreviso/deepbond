from deeptagger import opts
from deeptagger.dataset import fields


def build(options):
    prefixes_field = fields.AffixesField()
    suffixes_field = fields.AffixesField()
    caps_field = fields.CapsField()
    fields_tuples = []
    if options.use_prefixes:
        fields_tuples.append(('prefixes', prefixes_field))
    if options.use_suffixes:
        fields_tuples.append(('suffixes', suffixes_field))
    if options.use_caps:
        fields_tuples.append(('caps', caps_field))
    return fields_tuples


def load(path):
    options = opts.load(path)
    return build(options)


class Caps:
    all_upper = 'UPPER'  # acronyms
    all_lower = 'LOWER'  # normal words
    first_upper = 'FIRST'  # names, titles
    non_alpha = 'NON_ALPHA'  # dates, hours, punctuations
    other = 'OTHER'  # any other


def extract_prefixes(words, min_length, max_length):
    return extract_affixes(words, min_length, max_length, affix_type='prefix')


def extract_suffixes(words, min_length, max_length):
    return extract_affixes(words, min_length, max_length, affix_type='suffix')


def extract_affixes(words, min_length, max_length, affix_type='prefix'):
    total_length = max_length - min_length + 1
    pad_token = '<pad-{}>'.format(affix_type)

    def fill_with_pad(v):
        for _ in range(total_length - len(v)):
            v.append(pad_token)

    new_words = []
    for sentence in words:
        tokens = sentence.split()
        affixes_tokens = []
        for token in tokens:
            affixes = []
            if len(token) >= min_length:
                i, j = min_length, min(max_length, len(token))
                for k in range(i, j + 1):
                    affix = token[:k] if affix_type == 'prefix' else token[-k:]
                    affixes.append(affix)
            fill_with_pad(affixes)
            affixes_tokens.extend(affixes)
        new_words.append(' '.join(affixes_tokens))
    return new_words


def extract_caps(words):
    new_words = []
    for sentence in words:
        tokens = sentence.split()
        caps_tokens = []
        for token in tokens:
            if not token.isalpha():
                caps_tokens.append(Caps.non_alpha)
            elif token.isupper():
                caps_tokens.append(Caps.all_upper)
            elif token.islower():
                caps_tokens.append(Caps.all_lower)
            elif token[0].isupper() and token[1:].islower():
                caps_tokens.append(Caps.first_upper)
            else:
                caps_tokens.append(Caps.other)
        new_words.append(caps_tokens)
    return new_words
