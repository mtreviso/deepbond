import re
from pathlib import Path
import sys

punctuations = '!#$%&*+,-./:;<=>?@^|~'


def join_words_and_labels(words, labels):
    new_text = []
    assert len(words) == len(labels)
    for word, label in zip(words, labels):
        if label == '_':
            new_text.append(word)
        else:
            new_text.append(word)
            new_text.append(label)
    return new_text


def read_original_dir(dir_path):
    texts = []
    for f_path in sorted(dir_path.iterdir()):
        texts.append(read_original_file(f_path))
    return texts


def read_original_file(f_path):
    global punctuations
    with f_path.open('r', encoding='utf8') as f:
        text = f.read()
        text = re.sub(r'[%s]' % re.escape(punctuations), '', text)
        text = re.sub(r'\ +', ' ', text).strip()
        return text.split()


def write_labels(path, texts):
    dir_path = Path(path)
    assert len(list(dir_path.iterdir())) == len(texts)
    for f_path, orig_words in zip(sorted(dir_path.iterdir()), texts):
        f = f_path.open('r', encoding='utf8')
        labels = f.read().split()
        words_labels = join_words_and_labels(orig_words, labels)
        f.close()

        g = f_path.open('w', encoding='utf8')
        g.write(' '.join(words_labels))
        g.close()


if __name__ == '__main__':
    original_dir = sys.argv[1]
    predicted_dir = sys.argv[2]

    original_path = Path(original_dir)
    if original_path.is_dir():
        original_texts = read_original_dir(original_path)
    elif original_path.is_file():
        original_texts = read_original_file(original_path)
    else:
        raise Exception('You should inform a path to a dir or to a file.')
    write_labels(predicted_dir, original_texts)
