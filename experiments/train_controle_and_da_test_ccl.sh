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
					  --output-dir "runs/macmorpho-completo/" \
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
                      --save "saved-models/macmorpho-completo/" \
                      --tensorboard \


#--dev-path "data/transcriptions/ss/macmorpho_v1/dev.txt" \
