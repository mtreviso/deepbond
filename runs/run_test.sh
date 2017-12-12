#!/bin/bash

embpath="/media/mtreviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 10 -k 5 -b 1 --t bucket --models none rcnn --without-pos"

invokedb(){
	local ds=$1
	local et=$2
	local ed=$3
	local ea=$4
	myid="TEST_${ds}_${et}_${ed}_${ea}"
	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
	time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
}


# WORD2VEC
invokedb controle word2vec 50 sg

