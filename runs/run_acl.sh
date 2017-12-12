#!/bin/bash

embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 20 -k 5 -b 1 --t bucket --models rcnn none --without-pos"

invokedb(){
	local ds=$1
	local et=$2
	local ed=$3
	local ea=$4
	myid="ACL_${ds}_${et}_${ed}_${ea}"
	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
	time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
}


# WORD2VEC
# invokedb controle word2vec 50 sg
# invokedb controle word2vec 100 sg
# invokedb controle word2vec 300 sg
# invokedb controle word2vec 600 sg
# invokedb controle word2vec 50 cbow
# invokedb controle word2vec 100 cbow
# invokedb controle word2vec 300 cbow
# invokedb controle word2vec 600 cbow

# invokedb ccl word2vec 50 sg
# invokedb ccl word2vec 100 sg
# invokedb ccl word2vec 300 sg
# invokedb ccl word2vec 600 sg
# invokedb ccl word2vec 50 cbow
# invokedb ccl word2vec 100 cbow
# invokedb ccl word2vec 300 cbow
# invokedb ccl word2vec 600 cbow

# invokedb da word2vec 50 sg
# invokedb da word2vec 100 sg
# invokedb da word2vec 300 sg
# invokedb da word2vec 600 sg
# invokedb da word2vec 50 cbow
# invokedb da word2vec 100 cbow
# invokedb da word2vec 300 cbow
# invokedb da word2vec 600 cbow


# FASTTEXT
# invokedb controle fasttext 50 sg
# invokedb controle fasttext 100 sg
# invokedb controle fasttext 300 sg
invokedb controle fasttext 600 sg
invokedb controle fasttext 50 cbow
invokedb controle fasttext 100 cbow
invokedb controle fasttext 300 cbow
invokedb controle fasttext 600 cbow

invokedb ccl fasttext 50 sg
invokedb ccl fasttext 100 sg
invokedb ccl fasttext 300 sg
invokedb ccl fasttext 600 sg
invokedb ccl fasttext 50 cbow
invokedb ccl fasttext 100 cbow
invokedb ccl fasttext 300 cbow
invokedb ccl fasttext 600 cbow

invokedb da fasttext 50 sg
invokedb da fasttext 100 sg
invokedb da fasttext 300 sg
invokedb da fasttext 600 sg
invokedb da fasttext 50 cbow
invokedb da fasttext 100 cbow
invokedb da fasttext 300 cbow
invokedb da fasttext 600 cbow


# WANG2VEC
invokedb controle wang2vec 50 sg
invokedb controle wang2vec 100 sg
invokedb controle wang2vec 300 sg
invokedb controle wang2vec 600 sg
invokedb controle wang2vec 50 cbow
invokedb controle wang2vec 100 cbow
invokedb controle wang2vec 300 cbow
invokedb controle wang2vec 600 cbow

invokedb ccl wang2vec 50 sg
invokedb ccl wang2vec 100 sg
invokedb ccl wang2vec 300 sg
invokedb ccl wang2vec 600 sg
invokedb ccl wang2vec 50 cbow
invokedb ccl wang2vec 100 cbow
invokedb ccl wang2vec 300 cbow
invokedb ccl wang2vec 600 cbow

invokedb da wang2vec 50 sg
invokedb da wang2vec 100 sg
invokedb da wang2vec 300 sg
invokedb da wang2vec 600 sg
invokedb da wang2vec 50 cbow
invokedb da wang2vec 100 cbow
invokedb da wang2vec 300 cbow
invokedb da wang2vec 600 cbow


# GLOVE
# invokedb controle glove 50 none
# invokedb controle glove 100 none
# invokedb controle glove 300 none
# # invokedb controle glove 600 none

# invokedb ccl glove 50 none 
# invokedb ccl glove 100 none
# invokedb ccl glove 300 none
# # invokedb ccl glove 600 none

# invokedb da glove 50 none 
# invokedb da glove 100 none
# invokedb da glove 300 none
# invokedb da glove 600 none