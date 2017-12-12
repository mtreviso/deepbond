#!/bin/bash

# ss/dd_fillers/dd_editdisfs/ssdd
binary=""
task="dd_editdisfs${binary}"
wo_fillers="wo_fillers${binary}"


embpath="/home/treviso/Documents/saved-embeddings/ptbr"
et="word2vec"
ed="600"
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


# VER MELHOR RESULTADO E ADD PROSODIA
# TIREI OS FILLERS

# CTR
# invokemethod FEAT_ALL_EDITDISFS_CTR_RCNN_CRF 			"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"
# invokemethod FEAT_POS_EDITDISFS_CTR_RCNN_CRF 			"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb"
# invokemethod FEAT_EMB_EDITDISFS_CTR_RCNN_CRF 			"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos"
# invokemethod FEAT_HC_EDITDISFS_CTR_RCNN_CRF 			"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_HC_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
# invokemethod FEAT_EMB_AND_HC_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_EMB_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 

# HC E EMB FOI O MELHOR
# invokemethod FEAT_PROS_AND_ALL_EDITDISFS_CTR_RCNN_CRF 			"controle_editdisfs" rcnn_crf rcnn bucket 1 "--use-handcrafted"
# invokemethod FEAT_PROS_AND_POS_AND_HC_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs" rcnn_crf rcnn bucket 1 "--without-emb --use-handcrafted"

# CCL
# invokemethod FEAT_ALL_EDITDISFS_CCL_RCNN_CRF 			"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"
# invokemethod FEAT_POS_EDITDISFS_CCL_RCNN_CRF 			"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb"
# invokemethod FEAT_EMB_EDITDISFS_CCL_RCNN_CRF 			"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos"
# invokemethod FEAT_HC_EDITDISFS_CCL_RCNN_CRF 			"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_HC_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
# invokemethod FEAT_EMB_AND_HC_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_EMB_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 

# HC E EMB FOI O MELHOR
# invokemethod FEAT_PROS_AND_ALL_EDITDISFS_CCL_RCNN_CRF 			"ccl_editdisfs" rcnn_crf rcnn bucket 1 "--use-handcrafted"
# invokemethod FEAT_PROS_AND_POS_AND_HC_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs" rcnn_crf rcnn bucket 1 "--without-emb --use-handcrafted"


# DA
# invokemethod FEAT_ALL_EDITDISFS_DA_RCNN_CRF 		"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"
# invokemethod FEAT_POS_EDITDISFS_DA_RCNN_CRF 		"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb"
# invokemethod FEAT_EMB_EDITDISFS_DA_RCNN_CRF 		"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos"
# invokemethod FEAT_HC_EDITDISFS_DA_RCNN_CRF 			"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_HC_EDITDISFS_DA_RCNN_CRF 	"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-emb --use-handcrafted"
# invokemethod FEAT_EMB_AND_HC_EDITDISFS_DA_RCNN_CRF 	"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--without-pos --use-handcrafted"
# invokemethod FEAT_POS_AND_EMB_EDITDISFS_DA_RCNN_CRF "da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 

# HC E EMB FOI O MELHOR
# invokemethod FEAT_PROS_AND_ALL_EDITDISFS_DA_RCNN_CRF 			"da_editdisfs" rcnn_crf rcnn bucket 1 "--use-handcrafted"
# invokemethod FEAT_PROS_AND_POS_AND_HC_EDITDISFS_DA_RCNN_CRF 	"da_editdisfs" rcnn_crf rcnn bucket 1 "--without-emb --use-handcrafted"


# VERIFICANDO COM WORD2VEC
invokemethod WORD2VEC_FEAT_ALL_EDITDISFS_CTR_RCNN_CRF 	"controle_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"
invokemethod WORD2VEC_FEAT_ALL_EDITDISFS_DA_RCNN_CRF 	"da_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"
invokemethod WORD2VEC_FEAT_ALL_EDITDISFS_CCL_RCNN_CRF 	"ccl_editdisfs_${wo_fillers}" rcnn_crf none bucket 1 "--use-handcrafted"