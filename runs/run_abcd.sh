#!/bin/bash

# load ctr
# time sudo python3 -m deepbond --id ABCD_abcd_controle_fasttext_300_sg -d abcd_controle --emb-type fasttext --emb-file /media/treviso/SAMSUNG/Embeddings/ptbr/fasttext/pt_fasttext_sg_300.emb --save-predictions ABCD_abcd_controle_fasttext_300_sg --gpu -w 7 -e 20 -r 0.0 -b 1 --t bucket --models rcnn none --load  -k 0

# load ccl
# time sudo python3 -m deepbond --id ABCD_abcd_ccl_fasttext_300_sg -d abcd_ccl --emb-type fasttext --emb-file /media/treviso/SAMSUNG/Embeddings/ptbr/fasttext/pt_fasttext_sg_300.emb --save-predictions ABCD_abcd_ccl_fasttext_300_sg --gpu -w 7 -e 20 -r 0.0 -b 1 --t bucket --models rcnn none -k 0 --load


# embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
# fixedparams="--gpu -w 7 -e 20 -k 1 -r 0.0 -b 1 --t bucket --models rcnn none"

# invokedb(){
# 	local ds=$1
# 	local et=$2
# 	local ed=$3
# 	local ea=$4
# 	myid="ABCD_${ds}_${et}_${ed}_${ea}"
# 	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
# 	time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
# }

# invokedb abcd_controle fasttext 300 sg

