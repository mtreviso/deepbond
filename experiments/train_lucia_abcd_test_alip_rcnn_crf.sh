#!/usr/bin/env bash


python3 -m deepbond train \
                          --seed 42 \
                          --gpu-id 0  \
                          --output-dir "runs/test-cinderela-cachorro-test-alip/" \
                          --save "saved-models/test-cinderela-cachorro-test-alip/" \
                          --print-parameters-per-layer \
                          --final-report \
                          \
                          --train-path "data/transcriptions/Lucia_ABCD/CCL_Controle/" \
                          --dev-path "data/transcriptions/ALIP/test/" \
                          --punctuations ".?!"\
                          --max-length 9999999 \
                          --min-length 0 \
                          \
                          --vocab-size 9999999 \
                          --vocab-min-frequency 1 \
                          --keep-rare-with-vectors \
                          --add-embeddings-vocab \
                          \
                          --embeddings-format "word2vec" \
                          --embeddings-path "data/embs/word2vec/pt_word2vec_sg_600.kv.emb" \
                          --emb-dropout 0.0 \
                          --embeddings-binary \
                          --freeze-embeddings \
                          \
                          --model rcnn_crf \
                          \
                          --conv-size 100 \
                          --kernel-size 7 \
                          --pool-length 3 \
                          --cnn-dropout 0.0 \
                          \
                          --rnn-type lstm \
                          --hidden-size 100 \
                          --bidirectional \
                          --sum-bidir \
                          --rnn-dropout 0.5 \
                          \
                          --loss-weights "balanced" \
                          --train-batch-size 1 \
                          --dev-batch-size 1 \
                          --epochs 40 \
                          --optimizer "adamw" \
                          --learning-rate 0.001 \
                          --weight-decay 0.01 \
                          --save-best-only \
                          --early-stopping-patience 10 \
                          --restore-best-model


python3 -m deepbond predict \
                      --gpu-id 0  \
                      --prediction-type classes \
                      --load "saved-models/test-cinderela-cachorro-test-alip/" \
                      --test-path "data/transcriptions/ALIP/test/" \
                      --output-dir "data/transcriptions/ALIP/pred/"


python3 scripts/join_original_text_with_predicted_labels.py "data/transcriptions/ALIP/test/" "data/transcriptions/ALIP/pred/predictions"


python3 scripts/error_analysis.py "data/transcriptions/ALIP/test/*" "data/transcriptions/ALIP/pred/predictions/*"
