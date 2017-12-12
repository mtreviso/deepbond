#!/bin/bash

embpath="/media/mtreviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 15 -k 5 -b 1 --t bucket --models rcnn none --without-pos --save"

invokedb(){
	local ds=$1
	local et=$2
	local ed=$3
	local ea=$4
	myid="CACHORRO_${ds}_${et}_${ed}_${ea}"
	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
	time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
}


# WORD2VEC
# invokedb "cachorro_ccl" word2vec 600 sg

# invokedb "cachorro_da_analfabetos" word2vec 600 sg
# invokedb "cachorro_da_baixa" word2vec 600 sg

invokedb "cachorro_saudaveis_alta" word2vec 600 sg
# invokedb "cachorro_saudaveis_analfabetos" word2vec 600 sg
invokedb "cachorro_saudaveis_baixa" word2vec 600 sg

# invokedb "cachorro_da_all" word2vec 600 sg
invokedb "cachorro_saudaveis_all" word2vec 600 sg

invokedb "cachorro_all" word2vec 600 sg