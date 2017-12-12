# ss/dd_fillers/dd_editdisfs/ssdd

binary=""
task="dd_editdisfs${binary}"
wo_fillers="wo_fillers${binary}"

embpath="/home/treviso/Documents/saved-embeddings/ptbr"
ed="300"
ea="sg"

invokemethod(){	
	local myid=$1
	local ds=$2
	local ma=$3
	local mb=$4
	local st=$5
	local bs=$6
	local et=$7
	ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
	fixedparams="--gpu -w 7 -e 20 -k 5 --task ${task} --save --emb-type ${et} --emb-file ${ef} --without-pos"
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
}

# RCNN
invokemethod EMB_ONLY_WORD2VEC_EDITDISFS_NOFILLER_CTR_RCNN_CRF "controle_editdisfs_${wo_fillers}" 	rcnn_crf none bucket 1 word2vec
invokemethod EMB_ONLY_WORD2VEC_EDITDISFS_NOFILLER_CCL_RCNN_CRF "ccl_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 word2vec
invokemethod EMB_ONLY_WORD2VEC_EDITDISFS_NOFILLER_DA_RCNN_CRF  "da_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 word2vec

invokemethod EMB_ONLY_FASTTEXT_EDITDISFS_NOFILLER_CTR_RCNN_CRF "controle_editdisfs_${wo_fillers}" 	rcnn_crf none bucket 1 fasttext
invokemethod EMB_ONLY_FASTTEXT_EDITDISFS_NOFILLER_CCL_RCNN_CRF "ccl_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 fasttext
invokemethod EMB_ONLY_FASTTEXT_EDITDISFS_NOFILLER_DA_RCNN_CRF  "da_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 fasttext

invokemethod EMB_ONLY_WANG2VEC_EDITDISFS_NOFILLER_CTR_RCNN_CRF "controle_editdisfs_${wo_fillers}" 	rcnn_crf none bucket 1 wang2vec
invokemethod EMB_ONLY_WANG2VEC_EDITDISFS_NOFILLER_CCL_RCNN_CRF "ccl_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 wang2vec
invokemethod EMB_ONLY_WANG2VEC_EDITDISFS_NOFILLER_DA_RCNN_CRF  "da_editdisfs_${wo_fillers}" 		rcnn_crf none bucket 1 wang2vec
