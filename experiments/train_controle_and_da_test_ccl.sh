#!/usr/bin/env bash

python3 -m deepbond train \
                      --model rcnn \
                      --rnn-type gru \
                      --dropout 0.2 \
                      --hidden-size 200 \
                      --bidirectional \
                      --train-path "data/transcriptions/ss/Controle" \
					  --dev-path "data/transcriptions/ss/DA-Leve/" \
					  --test-path "data/transcriptions/ss/CCL-A/" \
					  --punctuations ".?!" \
					  --loss-weights "balanced" \
					  --train-batch-size 8 \
					  --dev-batch-size 8 \
					  --epochs 1 \
					  --optimizer "adam" \
					  --save-best-only \
					  --early-stopping-patience 2 \
					  --restore-best-model \
					  --final-report \
					  --keep-rare-with-vectors \
					  --add-embeddings-vocab \
					  --output-dir "runs/test-cinderela/" \
                      --save "saved-models/test-cinderela/" \
                      --tensorboard \
                      --use-conv \
                      --use-attention \
                      --use-linear \


#                      --embeddings-format "word2vec" \
#                      --embeddings-path "data/embeddings/word2vec/pt_word2vec_sg_600.kv.emb" \

exit;

python3 -m deepbond predict \
                      --load "saved-models/test-cinderela/" \
					  --prediction-type classes \
					  --output-dir "predictions/testing-cinderela/" \
					  --test-path "data/transcriptions/ss/CCL-A/" \
					  # --text "Há livros escritos para evitar espaços vazios na estante ."

