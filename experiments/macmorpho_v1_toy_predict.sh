#!/usr/bin/env bash
cd ..
python3 -m deeptagger predict \
                      --load "saved-models/testing-toy-save/" \
					  --prediction-type probas \
					  --output-dir "predictions/macmorpho_v1_toy/" \
					  --test-path "data/corpus/pt/macmorpho_v1_toy/test.txt" \
					  # --text "Há livros escritos para evitar espaços vazios na estante ."
