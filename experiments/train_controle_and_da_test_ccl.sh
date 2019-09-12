#!/usr/bin/env bash

python3 -m deepbond train \
                      --model rcnn \
                      --rnn-type lstm \
                      --dropout 0.2 \
                      --hidden-size 200 \
                      --bidirectional \
                      --train-path "data/transcriptions/ss/Controle" \
					  --test-path "data/transcriptions/ss/CCL-A/" \
					  --punctuations ".?!" \
					  --loss-weights "balanced" \
					  --embeddings-format "word2vec" \
					  --embeddings-path "data/embeddings/word2vec/pt_word2vec_sg_600.kv.emb" \
					  --output-dir "runs/test-cinderela/" \
					  --train-batch-size 8 \
					  --dev-batch-size 8 \
					  --epochs 5 \
					  --optimizer "adam" \
					  --save-best-only \
					  --early-stopping-patience 2 \
					  --restore-best-model \
					  --final-report \
					  --add-embeddings-vocab \
					  --keep-rare-with-vectors \
					  --add-embeddings-vocab \
                      --save "saved-models/test-cinderela/" \
                      --tensorboard \


#--dev-path "data/transcriptions/ss/macmorpho_v1/dev.txt" \
