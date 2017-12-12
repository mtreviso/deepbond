#!/bin/bash


invokemet(){
	local ds=$1
	local emb=$2
	# time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
	python3 error_analysis.py $ds $emb | grep "NIST" -m 1
}

invokemet ../data/corpus/Controle/ ../data/saves/MET_MLP_CTR
invokemet ../data/corpus/Controle/ ../data/saves/MET_CRF_CTR
invokemet ../data/corpus/Controle/ ../data/saves/MET_RNN_CTR
invokemet ../data/corpus/Controle/ ../data/saves/MET_CNN_CTR
invokemet ../data/corpus/Controle/ ../data/saves/MET_RCNN_CTR
echo "---"
invokemet ../data/corpus/CCL-A/ ../data/saves/MET_CRF_CCL
invokemet ../data/corpus/CCL-A/ ../data/saves/MET_MLP_CCL
invokemet ../data/corpus/CCL-A/ ../data/saves/MET_RNN_CCL
invokemet ../data/corpus/CCL-A/ ../data/saves/MET_CNN_CCL
invokemet ../data/corpus/CCL-A/ ../data/saves/MET_RCNN_CCL
echo "---"
invokemet ../data/corpus/Constituicao-Filtrada-Artigos/ ../data/saves/MET_CRF_CONST
invokemet ../data/corpus/Constituicao-Filtrada-Artigos/ ../data/saves/MET_MLP_CONST
invokemet ../data/corpus/Constituicao-Filtrada-Artigos/ ../data/saves/MET_RNN_CONST
invokemet ../data/corpus/Constituicao-Filtrada-Artigos/ ../data/saves/MET_CNN_CONST
invokemet ../data/corpus/Constituicao-Filtrada-Artigos/ ../data/saves/MET_RCNN_CONST
echo "---"
invokemet ../data/corpus/Constituicao-Mini-Filtrada-Artigos/ ../data/saves/MET_CRF_CONST_MINI
invokemet ../data/corpus/Constituicao-Mini-Filtrada-Artigos/ ../data/saves/MET_MLP_CONST_MINI
invokemet ../data/corpus/Constituicao-Mini-Filtrada-Artigos/ ../data/saves/MET_RNN_CONST_MINI
invokemet ../data/corpus/Constituicao-Mini-Filtrada-Artigos/ ../data/saves/MET_CNN_CONST_MINI
invokemet ../data/corpus/Constituicao-Mini-Filtrada-Artigos/ ../data/saves/MET_RCNN_CONST_MINI


