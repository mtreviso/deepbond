#!/bin/bash


conferencia="BRACIS"
embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 10 -k 5 -b 1 --t bucket --models rcnn none --without-pos"


invokedb(){
	local ds=$1
	local et=$2
	local ed=$3
	local ea=$4
	myid="${conferencia}_${ds}_${et}_${ed}_${ea}"
	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
	time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
}


# WORD2VEC
# invokedb constituicao_mini word2vec 50 sg
# invokedb constituicao_mini word2vec 100 sg
# invokedb constituicao_mini word2vec 300 sg
# invokedb constituicao_mini word2vec 600 sg
# invokedb constituicao_mini word2vec 50 cbow
# invokedb constituicao_mini word2vec 100 cbow
# invokedb constituicao_mini word2vec 300 cbow
# invokedb constituicao_mini word2vec 600 cbow

# invokedb constituicao word2vec 50 sg
# invokedb constituicao word2vec 100 sg
# invokedb constituicao word2vec 300 sg
# invokedb constituicao word2vec 600 sg
# invokedb constituicao word2vec 50 cbow
# invokedb constituicao word2vec 100 cbow
# invokedb constituicao word2vec 300 cbow
# invokedb constituicao word2vec 600 cbow



# # FASTTEXT
# invokedb constituicao_mini fasttext 50 sg
# invokedb constituicao_mini fasttext 100 sg
# invokedb constituicao_mini fasttext 300 sg
# invokedb constituicao_mini fasttext 600 sg
# invokedb constituicao_mini fasttext 50 cbow
# invokedb constituicao_mini fasttext 100 cbow
# invokedb constituicao_mini fasttext 300 cbow
# invokedb constituicao_mini fasttext 600 cbow

# invokedb constituicao fasttext 50 sg
# invokedb constituicao fasttext 100 sg
# invokedb constituicao fasttext 300 sg
# invokedb constituicao fasttext 600 sg
# invokedb constituicao fasttext 50 cbow
# invokedb constituicao fasttext 100 cbow
# invokedb constituicao fasttext 300 cbow
# invokedb constituicao fasttext 600 cbow



# # WANG2VEC
# invokedb constituicao_mini wang2vec 50 sg
# invokedb constituicao_mini wang2vec 100 sg
# invokedb constituicao_mini wang2vec 300 sg
# invokedb constituicao_mini wang2vec 600 sg
# invokedb constituicao_mini wang2vec 50 cbow
# invokedb constituicao_mini wang2vec 100 cbow
# invokedb constituicao_mini wang2vec 300 cbow
# invokedb constituicao_mini wang2vec 600 cbow

# invokedb constituicao wang2vec 50 sg
# invokedb constituicao wang2vec 100 sg
# invokedb constituicao wang2vec 300 sg
invokedb constituicao wang2vec 600 sg
invokedb constituicao wang2vec 50 cbow
invokedb constituicao wang2vec 100 cbow
invokedb constituicao wang2vec 300 cbow
invokedb constituicao wang2vec 600 cbow



