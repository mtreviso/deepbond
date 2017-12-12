# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_fillers"

embpath="/home/treviso/Documents/saved-embeddings/ptbr"
et="fasttext"
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
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
}


# RCNN EH
invokemethod ZZZ_FILLERS_EH_CTR_RCNN controle_fillers_eh rcnn rcnn bucket 1
invokemethod ZZZ_FILLERS_EH_CCL_RCNN ccl_fillers_eh rcnn rcnn bucket 1
invokemethod ZZZ_FILLERS_EH_DA_RCNN da_fillers_eh rcnn none bucket 1

# RCNN NORMAL
invokemethod ZZZ_FILLERS_SEMEH_CTR_RCNN controle_fillers rcnn rcnn bucket 1
invokemethod ZZZ_FILLERS_SEMEH_CCL_RCNN ccl_fillers rcnn rcnn bucket 1
invokemethod ZZZ_FILLERS_SEMEH_DA_RCNN da_fillers rcnn none bucket 1
