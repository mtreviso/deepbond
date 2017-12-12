#!/bin/bash

# ss/dd_fillers/dd_editdisfs/ssdd
binary=""
task="dd_editdisfs${binary}"
wo_fillers="wo_fillers_eh${binary}"

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


# CTR
invokemethod METHOD_NOFILLER_EH_POS_AND_HC_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
# CCL
invokemethod METHOD_NOFILLER_EH_POS_AND_HC_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
# DA
invokemethod METHOD_NOFILLER_EH_POS_AND_HC_EDITDISFS_DA_RCNN_CRF 	"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
	