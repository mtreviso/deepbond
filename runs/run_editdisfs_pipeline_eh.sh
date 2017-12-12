#!/bin/bash

# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_editdisfs"

embpath="/home/treviso/Documents/saved-embeddings/ptbr"
et="wang2vec"
ed="300"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
fixedparams="--gpu -w 7 -e 20 -k 5 --task ${task} --save --emb-type ${et} --emb-file ${ef}"

invokemethod(){
	local myid=$1
	local ds=$2
	local ma=$3
	local mb=$4
	local st=$5
	local bs=$6
	local ex=$7
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams $ex
}


invokemethod PIPELINE_EH_CTR_RCNN_CRF 	"controle_pipeline" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"

invokemethod PIPELINE_EH_CCL_RCNN_CRF 	"ccl_pipeline" 		rcnn_crf none bucket 1 "--without-emb --use-handcrafted"

invokemethod PIPELINE_EH_DA_RCNN_CRF 	"da_pipeline" 		rcnn_crf none bucket 1 "--without-emb --use-handcrafted"

