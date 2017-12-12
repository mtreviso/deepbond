# python3 -m deepbond --id ControleRCNNMAX --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 32 --models rcnnmax crf
# python3 -m deepbond --id CCLRCNNMAX --gpu -d ccl -r 0.6 -w 7 -e 20 -k 5 -b 32 --models rcnnmax crf
# python3 -m deepbond --id CONSTMINIRCNNMAX_ALL --gpu -d constituicao_mini -r 0.85 -w 7 -e 20 -k 5 -b 32 --models rcnnmax rcnn mlp
# python3 -m deepbond --id CONSTMINIRCNNMAX_ALL --gpu -d constituicao_mini -r 0.85 -w 7 -e 20 -k 5 -b 32 --models crf
# python3 -m deepbond --id CONSTMINIRCNNMAX_NEMB --gpu -d constituicao_mini --embedding-type id -r 0.85 -w 7 -e 20 -k 5 -b 32 --models mlp
# python3 -m deepbond --id CONSTMINIRCNNMAX_NPOS --gpu -d constituicao_mini -r 0.85 -w 7 -e 20 -k 5 -b 32 --models rcnn mlp

# python3 -m deepbond --id PUCRS_FMUSP_ALL_OK_MLP --gpu -d pucrs_usp_controle_ccl -r 0 -w 7 -e 20 -k 3 -b 32 --models mlp
# python3 -m deepbond --id PUCRS_CONST_ALL_OK_MLP --gpu -d pucrs_constituicao_controle_ccl -r 0 -w 7 -e 20 -k 3 -b 32 --models mlp

# python3 -m deepbond --id PUCRS_FMUSP_CTR_OK_MLP --gpu -d pucrs_usp_controle -r 0 -w 7 -e 20 -k 3 -b 32 --models mlp
# python3 -m deepbond --id PUCRS_FMUSP_CCL_MLP --gpu -d pucrs_usp_ccl -r 0 -w 7 -e 20 -k 3 -b 32 --models mlp

# python3 -m deepbond --id PUCRS_CONST_CTR_OK_MLP --gpu -d pucrs_constituicao_controle -r 0 -w 7 -e 20 -k 3 -b 32 --models mlp
# python3 -m deepbond --id PUCRS_CONST_CCL_MLP --gpu -d pucrs_constituicao_ccl -r 0 -w 7 -e 20 -k 1 -b 32 --models mlp

# sudo python3 -m deepbond --id TestingPad --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax
# sudo python3 -m deepbond --id TestingPad --gpu -d ccl -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax



##################
# EACL
##################

# sudo python3 -m deepbond --id CONTROLE_ALL --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf mlp svm
# sudo python3 -m deepbond --id CCL_ALL --gpu -d ccl -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf mlp svm

# sudo python3 -m deepbond --id CONTROLE_ALL --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 4 --models pad_rcnn
# sudo python3 -m deepbond --id CCL_ALL 	   --gpu -d ccl 	 -r 0.6 -w 7 -e 20 -k 5 -b 4 --models pad_rcnn

# sudo python3 -m deepbond --id CONTROLE_NEMB --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 1 --emb-type id --models rcnnmax crf mlp
# sudo python3 -m deepbond --id CCL_NEMB 		--gpu -d ccl 	  -r 0.6 -w 7 -e 20 -k 5 -b 1 --emb-type id --models rcnnmax crf mlp

# sudo python3 -m deepbond --id CONTROLE_NPOS --gpu -d controle -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf mlp
# sudo python3 -m deepbond --id CCL_NPOS 		--gpu -d ccl 	  -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf mlp

# sudo python3 -m deepbond --id CONST_CNN 		--gpu -d constituicao 		-r 0.85 -w 7 -e 20 -k 5 -b 1 --models just_cnn
# sudo python3 -m deepbond --id CONST_MINI_CNN 	--gpu -d constituicao_mini 	-r 0.85 -w 7 -e 20 -k 5 -b 1 --models just_cnn
# sudo python3 -m deepbond --id CONST_RNN 		--gpu -d constituicao 		-r 0.85 -w 7 -e 20 -k 5 -b 1 --models just_rnn
# sudo python3 -m deepbond --id CONST_MINI_RNN 	--gpu -d constituicao_mini 	-r 0.85 -w 7 -e 20 -k 5 -b 1 --models just_rnn


##################
# EACL EXTRA
##################

# sudo python3 -m deepbond --id CONTROLE_X_ALL --gpu -d controle -r 0.0 -w 7 -e 20 -k 5 -b 1 --t bucket --models rcnn rcnn
# sudo python3 -m deepbond --id CCL_X_ALL 	 --gpu -d ccl -r 0.0 -w 7 -e 20 -k 5 -b 1 --t bucket --models rcnn rcnn

sudo python3 -m deepbond --id CONST_CTL --gpu -d constituicao_controle -r 0.0 -w 7 -e 20 -k 5 -b 1 --t bucket --models rcnn rcnn
sudo python3 -m deepbond --id CONST_CCL --gpu -d constituicao_ccl -r 0.0 -w 7 -e 20 -k 5 -b 1 --t bucket --models rcnn rcnn



# EMBEDDINGS:

# sudo python3 -m deepbond --id EMB_WORD2VEC_CCL_SG 	--gpu -d ccl --emb-type word2vec --emb-file data/embeddings/word2vec/wiki_g1_plnbr.sg.w2v 		 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp
# sudo python3 -m deepbond --id EMB_WORD2VEC_CCL_CBOW --gpu -d ccl --emb-type word2vec --emb-file data/embeddings/word2vec/wiki_g1_plnbr.w2v 	 		 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp

# sudo python3 -m deepbond --id EMB_GLOVE_CCL 		--gpu -d ccl --emb-type glove 	 --emb-file data/embeddings/glove/wiki_g1_plnbr.glove 			 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp

# sudo python3 -m deepbond --id EMB_FASTTEXT_CCL_SG 	--gpu -d ccl --emb-type fasttext --emb-file data/embeddings/fasttext/wiki_g1_plnbr.ft.sg.bin 	 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp
# sudo python3 -m deepbond --id EMB_FASTTEXT_CCL_CBOW --gpu -d ccl --emb-type fasttext --emb-file data/embeddings/fasttext/wiki_g1_plnbr.ft.cbow.bin 	 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp

# sudo python3 -m deepbond --id EMB_WANG2VEC_CCL_SG 	--gpu -d ccl --emb-type wang2vec --emb-file data/embeddings/wang2vec/wiki_g1_plnbr.sg.wang2vec 	 -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp
# sudo python3 -m deepbond --id EMB_WANG2VEC_CCL_CBOW --gpu -d ccl --emb-type wang2vec --emb-file data/embeddings/wang2vec/wiki_g1_plnbr.cbow.wang2vec -r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax mlp



# OUTPUT EVALUATION:

# sudo python3 -m deepbond --id CONTROLE_SAVE --gpu -d controle 	-r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf --save-predictions controle_eacl
# sudo python3 -m deepbond --id CCL_SAVE 		--gpu -d ccl 		-r 0.6 -w 7 -e 20 -k 5 -b 1 --models rcnnmax crf --save-predictions ccl_eacl