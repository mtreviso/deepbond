#!/usr/bin/env bash
cd ..
python3 -m deeptagger train \
                      --model simple_lstm \
                      --bidirectional \
                      --train-path "data/corpus/pt/macmorpho_v1_toy/train.txt" \
					  --dev-path "data/corpus/pt/macmorpho_v1_toy/dev.txt" \
					  --test-path "data/corpus/pt/macmorpho_v1_toy/test.txt" \
					  --del-word " " \
					  --del-tag "_" \
					  --embeddings-format "polyglot" \
					  --embeddings-path "data/embeddings/polyglot/pt/embeddings_pkl.tar.bz2" \
					  --output-dir "runs/testing-macmorpho_v1_toy/" \
					  --train-batch-size 128 \
					  --dev-batch-size 128 \
					  --optimizer adam \
					  --save-best-only \
					  --early-stopping-patience 3 \
					  --restore-best-model \
					  --final-report \
					  --add-embeddings-vocab \
					  --epochs 2 \
					  --load "saved-models/testing-toy-save-simple-lstm/" \
					  --use-prefixes \
					  --use-suffixes \
					  --use-caps


