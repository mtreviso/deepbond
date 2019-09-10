# lowercased special tokens
UNK = '<unk>'
PAD = '<pad>'
START = '<bos>'
STOP = '<eos>'

# special tokens id (don't edit this order)
UNK_ID = 0
PAD_ID = 1

# this should be set later after building fields
TAGS_PAD_ID = 0
NB_LABELS = 0

# output_dir
OUTPUT_DIR = 'runs'

# default filenames
CONFIG = 'config.json'
DATASET = 'dataset.torch'
MODEL = 'model.torch'
OPTIMIZER = 'optim.torch'
SCHEDULER = 'scheduler.torch'
TRAINER = 'trainer.torch'
VOCAB = 'vocab.torch'
PREDICTIONS = 'predictions.txt'
