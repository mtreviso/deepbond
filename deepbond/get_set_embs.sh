#!/bin/bash


invokeemb(){
	local ds=$1
	local emb=$2
	# time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
	python3 error_analysis.py $ds $emb | grep NIST -m 1
}


invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_50_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_100_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_300_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_600_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_50_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_100_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_300_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_fasttext_600_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_50_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_100_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_300_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_600_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_50_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_100_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_300_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_wang2vec_600_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_50_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_100_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_300_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_600_sg
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_50_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_100_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_300_cbow
invokeemb ../data/corpus/CCL-A/ ../data/saves/ACL_ccl_word2vec_600_cbow
echo "---"
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_50_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_100_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_300_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_600_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_50_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_100_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_300_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_fasttext_600_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_50_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_100_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_300_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_600_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_50_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_100_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_300_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_wang2vec_600_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_50_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_100_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_300_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_600_sg
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_50_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_100_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_300_cbow
invokeemb ../data/corpus/Controle/ ../data/saves/ACL_controle_word2vec_600_cbow
echo "---"
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_50_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_100_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_300_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_600_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_50_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_100_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_300_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_fasttext_600_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_50_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_100_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_300_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_600_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_50_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_100_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_300_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_wang2vec_600_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_50_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_100_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_300_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_600_sg
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_50_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_100_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_300_cbow
invokeemb ../data/corpus/DA-Leve/ ../data/saves/ACL_da_word2vec_600_cbow
