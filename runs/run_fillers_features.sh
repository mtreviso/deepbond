# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_fillers"

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


# CTR
# invokemethod FEAT_ALL_FILLERS_CTR_RCNN controle_fillers rcnn rcnn bucket 1
invokemethod FEAT_NOEMB_FILLERS_CTR_RCNN controle_fillers rcnn rcnn bucket 1 --without-emb
invokemethod FEAT_NOPOS_FILLERS_CTR_RCNN controle_fillers rcnn rcnn bucket 1 --without-pos


# CCL
# invokemethod FEAT_ALL_FILLERS_CCL_RCNN ccl_fillers rcnn rcnn bucket 1
invokemethod FEAT_NOEMB_FILLERS_CCL_RCNN ccl_fillers rcnn rcnn bucket 1 --without-emb
invokemethod FEAT_NOPOS_FILLERS_CCL_RCNN ccl_fillers rcnn rcnn bucket 1 --without-pos


# DA
# invokemethod FEAT_ALL_FILLERS_DA_RCNN da_fillers rcnn none bucket 1
invokemethod FEAT_NOEMB_FILLERS_DA_RCNN da_fillers rcnn none bucket 1 --without-emb
invokemethod FEAT_NOPOS_FILLERS_DA_RCNN da_fillers rcnn none bucket 1 --without-pos

