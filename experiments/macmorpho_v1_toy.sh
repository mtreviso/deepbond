#!/usr/bin/env bash
cd ..
python3 -m deeptagger train \
                      --model "rcnn" \
                      --rnn-type "gru" \
                      --bidirectional \
                      --train-path "data/corpus/pt/macmorpho_v1_toy/train.txt" \
                      --dev-path "data/corpus/pt/macmorpho_v1_toy/dev.txt" \
                      --test-path "data/corpus/pt/macmorpho_v1_toy/test.txt" \
                      --del-word " " \
                      --del-tag "_" \
                      --output-dir "runs/macmorpho_v1_toy_rcnn_embs/" \
                      --train-batch-size 128 \
                      --dev-batch-size 128 \
                      --optimizer "adam" \
                      --save-best-only \
                      --early-stopping-patience 3 \
                      --restore-best-model \
                      --final-report \
                      --epochs 2 \
                      --use-prefixes \
                      --use-suffixes \
                      --use-caps \
                      --save "saved-models/macmorpho_v1_toy_rcnn_ebs/" \
                      --embeddings-format "fonseca" \
                      --embeddings-path "data/pretrained-embeddings/fonseca/"
                      # --tensorboard \
                      # --amsgrad \
                      # --embeddings-format "fasttext" \
                      # --embeddings-path "data/pretrained-embeddings/fasttext/skip_s300.txt" \
                      # --add-embeddings-vocab \
                      # --keep-rare-with-vectors
                      # --lr-step-decay noam \
                      # --nesterov \
                      # --momentum 0.9 \
                      # --warmup-steps 2000 \
                      # --scheduler "exponential" \
                      # --gamma 0.1 \
                      
