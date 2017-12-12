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
	fixedparams="--gpu -w 7 -e 20 -k 5 --task ${task} --save --emb-type ${et} --emb-file ${ef} --use-handcrafted"
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
}

# RCNN
invokemethod EMB_WORD2VEC_EDITDISFSWOF_CTR_RCNN "controle_editdisfs_${wo_fillers}" mlp none window 32 word2vec
# invokemethod EMB_WORD2VEC_EDITDISFSWOF_CCL_RCNN "ccl_editdisfs_${wo_fillers}" mlp none window 32 word2vec
# invokemethod EMB_WORD2VEC_EDITDISFSWOF_DA_RCNN "da_editdisfs_${wo_fillers}" mlp none window 32 word2vec

# invokemethod EMB_FASTTEXT_EDITDISFSWOF_CTR_RCNN "controle_editdisfs_${wo_fillers}" mlp none window 32 fasttext
# invokemethod EMB_FASTTEXT_EDITDISFSWOF_CCL_RCNN "ccl_editdisfs_${wo_fillers}" mlp none window 32 fasttext
# invokemethod EMB_FASTTEXT_EDITDISFSWOF_DA_RCNN "da_editdisfs_${wo_fillers}" mlp none window 32 fasttext

# invokemethod EMB_WANG2VEC_EDITDISFSWOF_CTR_RCNN "controle_editdisfs_${wo_fillers}" mlp none window 32 wang2vec
# invokemethod EMB_WANG2VEC_EDITDISFSWOF_CCL_RCNN "ccl_editdisfs_${wo_fillers}" mlp none window 32 wang2vec
# invokemethod EMB_WANG2VEC_EDITDISFSWOF_DA_RCNN "da_editdisfs_${wo_fillers}" mlp none window 32 wang2vec
