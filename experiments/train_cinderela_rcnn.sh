#!/usr/bin/env bash


#python3 -m deepbond train \
#                          --seed 42 \
#                          --gpu-id 0  \
#                          --output-dir "runs/rcnn-cinderela-all/" \
#                          --save "saved-models/rcnn-cinderela-all/" \
#                          --print-parameters-per-layer \
#                          --final-report \
#                          \
#                          --train-path "data/transcriptions/Cinderela_ALL/" \
#                          --punctuations ".?!"\
#                          --max-length 9999999 \
#                          --min-length 0 \
#                          \
#                          --vocab-size 9999999 \
#                          --vocab-min-frequency 1 \
#                          --keep-rare-with-vectors \
#                          --add-embeddings-vocab \
#                          \
#                          --emb-dropout 0.0 \
#                          --embeddings-format "text" \
#                          --embeddings-path "data/embs/word2vec/pt_word2vec_sg_600.small.pickle.emb" \
#                          --embeddings-binary \
#                          --freeze-embeddings \
#                          \
#                          --model rcnn \
#                          \
#                          --conv-size 100 \
#                          --kernel-size 7 \
#                          --pool-length 3 \
#                          --cnn-dropout 0.0 \
#                          \
#                          --rnn-type lstm \
#                          --hidden-size 100 \
#                          --bidirectional \
#                          --sum-bidir \
#                          --rnn-dropout 0.5 \
#                          \
#                          --loss-weights "balanced" \
#                          --train-batch-size 4 \
#                          --dev-batch-size 4 \
#                          --epochs 10 \
#                          --optimizer "adamw" \
#                          --learning-rate 0.001 \
#                          --weight-decay 0.01 \
#                          --save-best-only \
#                          --early-stopping-patience 10 \
#                          --restore-best-model


#dirpath="Cachorro/CCL Alta"
#dirpath="Cachorro/CCL Analfabetos"
#dirpath="Cachorro/CCL Baixa"
#dirpath="Cachorro/DA Analfabetos"
#dirpath="Cachorro/DA Baixa"
#dirpath="Cachorro/Saudáveis Alta"
#dirpath="Cachorro/Saudáveis Analfabetos"
dirpath="Cachorro/Saudáveis Baixa"
python3 -m deepbond predict \
                      --gpu-id 0  \
                      --prediction-type classes \
                      --load "saved-models/rcnn-cinderela-all/" \
                      --test-path "data/transcriptions/${dirpath}/" \
                      --output-dir "data/predictions/${dirpath}/"

# Merge text and predictions
python3 scripts/join_original_text_with_predicted_labels.py "data/transcriptions/${dirpath}/" "data/predictions/${dirpath}/predictions/"

# Error analysis
python3 scripts/error_analysis.py "data/transcriptions/${dirpath}/*" "data/predictions/${dirpath}/predictions/*"
