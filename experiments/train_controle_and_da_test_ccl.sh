#!/usr/bin/env bash

python3 -m deepbond train \
                      --seed 42 \
					  --output-dir "runs/test-cinderela-togo/" \
                      --save "saved-models/test-cinderela-togo/" \
                      --tensorboard \
                      --final-report \
                      \
                      --train-path "data/transcriptions/folds/CCL-A/0/train/" \
					  --dev-path "data/transcriptions/folds/CCL-A/0/test/" \
					  --punctuations ".?!" \
					  \
                      --max-length 9999999 \
                      --min-length 0 \
                      \
                      --vocab-size 9999999 \
                      --vocab-min-frequency 1 \
                      --keep-rare-with-vectors \
					  --add-embeddings-vocab \
                      \
                      --model rcnn \
                      \
                      --emb-dropout 0 \
                      \
                      --use-conv \
                      --conv-size 100 \
                      --kernel-size 7 \
                      --pool-length 3 \
                      \
                      --use-rnn \
                      --rnn-type gru \
                      --hidden-size 200 \
                      --bidirectional \
                      --sum-bidir \
                      --dropout 0.5 \
                      \
                      --use-linear \
                      \
					  --loss-weights "balanced" \
					  --train-batch-size 2 \
					  --dev-batch-size 2 \
					  --epochs 20 \
					  --optimizer "adam" \
					  --save-best-only \
					  --early-stopping-patience 5 \
					  --restore-best-model \

python3 -m deepbond predict \
                      --prediction-type classes \
                      --load "saved-models/test-cinderela-togo/" \
					  --test-path "data/transcriptions/folds/CCL-A/0/test/" \
					  --output-dir "data/transcriptions/folds/CCL-A/0/pred/" \
					  # --text "Há livros escritos para evitar espaços vazios na estante ."


python3 scripts/join_original_text_with_predicted_labels.py data/transcriptions/folds/CCL-A/0/train/ data/transcriptions/folds/CCL-A/0/pred/predictions/
