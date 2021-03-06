#!/usr/bin/env bash

python3 -m deepbond train \
                      --gpu-id 0 \
                      --seed 42 \
					            --output-dir "runs/test-cinderela/" \
                      --save "saved-models/test-cinderela/" \
                      --print-parameters-per-layer \
                      --final-report \
                      \
                      --train-path "data/folds/CCL-A/0/train/" \
					            --dev-path "data/folds/CCL-A/0/test/" \
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
                      --model cnn_attn \
                      --attn-type 'multihead' \
                      --attn-scorer 'general' \
                      --attn-hidden-size 200 \
                      --attn-dropout 0.0 \
                      --attn-nb-heads 4 \
                      --attn-multihead-hidden-size 136 \
                      \
                      --emb-dropout 0 \
                      --freeze-embeddings \
                      \
                      --conv-size 100 \
                      --kernel-size 7 \
                      --pool-length 3 \
                      --cnn-dropout 0.0 \
                      \
                      --rnn-type gru \
                      --hidden-size 200 \
                      --bidirectional \
                      --sum-bidir \
                      --rnn-dropout 0.7 \
                      \
                      --loss-weights "balanced" \
                      --train-batch-size 1 \
                      --dev-batch-size 1 \
                      --epochs 2 \
                      --optimizer "adamw" \
                      --learning-rate 0.001 \
                      --weight-decay 0.01 \
                      --save-best-only \
                      --early-stopping-patience 6 \
                      --restore-best-model \

#--embeddings-format "word2vec" \
#--embeddings-path "data/embeddings/word2vec/pt_word2vec_sg_600.kv.emb" \

python3 -m deepbond predict \
                      --gpu-id 0 \
                      --prediction-type classes \
                      --load "saved-models/test-cinderela/" \
                      --test-path "data/folds/CCL-A/0/test/" \
                      --output-dir "data/folds/CCL-A/0/pred/" \
                      --punctuations ".?!" \
                      # --text "Há livros escritos para evitar espaços vazios na estante ."


python3 scripts/join_original_text_with_predicted_labels.py data/folds/CCL-A/0/test/ data/folds/CCL-A/0/pred/predictions/
