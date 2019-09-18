from pathlib import Path
import sys


def join_words_and_labels(words, labels):
    new_text = []
    for word, label in zip(words, labels):
        if label == '_':
            new_text.append(word)
        else:
            new_text.append(word)
            new_text.append(label)
    return new_text


def read_original_dir(path):
    texts = []
    dir_path = Path(path)
    for f_path in sorted(dir_path.iterdir()):
        with f_path.open('r', encoding='utf8') as f:
            texts.append(f.read().split())
    return texts


def write_labels(path, texts):
    dir_path = Path(path)
    for orig_words, f_path in zip(texts, sorted(dir_path.iterdir())):
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

    original_texts = read_original_dir(original_dir)
    write_labels(predicted_dir, original_texts)
