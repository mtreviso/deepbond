# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_fillers"

embpath="/home/treviso/Documents/saved-embeddings/ptbr"
et="word2vec"
ed="600"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
fixedparams="--gpu -w 7 -e 25 -k 5 --task ${task} --save --emb-type ${et} --emb-file ${ef}"

invokemethod(){
	local myid=$1
	local ds=$2
	local ma=$3
	local mb=$4
	local st=$5
	local bs=$6
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
}


# invokemethod PROSODIC_FILLERS_CTR_CRF controle_fillers 	none crf dicted 1
# invokemethod PROSODIC_FILLERS_CCL_CRF ccl_fillers 		none crf dicted 1
# invokemethod PROSODIC_FILLERS_CTR_MLP controle_fillers 	none mlp window 32
# invokemethod PROSODIC_FILLERS_CCL_MLP ccl_fillers 		none mlp window 32
# invokemethod PROSODIC_FILLERS_CTR_CNN controle_fillers 	none cnn bucket 1
# invokemethod PROSODIC_FILLERS_CCL_CNN ccl_fillers 		none cnn bucket 1
# invokemethod PROSODIC_FILLERS_CTR_RCNN controle_fillers none rcnn bucket 1
# invokemethod PROSODIC_FILLERS_CCL_RCNN ccl_fillers 		none rcnn bucket 1
# invokemethod PROSODIC_FILLERS_CTR_RNN controle_fillers 	none rnn bucket 1
# invokemethod PROSODIC_FILLERS_CCL_RNN ccl_fillers 		none rnn bucket 1


# CRF
invokemethod OK_FILLERS_NEW_CTR_CRF controle_fillers crf crf dicted 1
invokemethod OK_FILLERS_NEW_CCL_CRF ccl_fillers crf crf dicted 1

# MLP
# invokemethod FILLERS_CTR_MLP controle_fillers mlp mlp window 32
# invokemethod FILLERS_CCL_MLP ccl_fillers mlp mlp window 32

# CNN
# invokemethod FILLERS_CTR_CNN controle_fillers cnn cnn bucket 1
# invokemethod FILLERS_CCL_CNN ccl_fillers cnn cnn bucket 1

# RCNN
# invokemethod FILLERS_CTR_RCNN controle_fillers rcnn rcnn bucket 1
# invokemethod FILLERS_CCL_RCNN ccl_fillers rcnn rcnn bucket 1

# RNN
# invokemethod FILLERS_CTR_RNN controle_fillers rnn rnn bucket 1
# invokemethod FILLERS_CCL_RNN ccl_fillers rnn rnn bucket 1


# DA patients (none for audio data)
invokemethod OK_FILLERS_DA_CRF da_fillers crf none dicted 1
# invokemethod FILLERS_DA_MLP da_fillers mlp none window 32
# invokemethod FILLERS_DA_CNN da_fillers cnn none bucket 1
# invokemethod FILLERS_DA_RCNN da_fillers rcnn none bucket 1
# invokemethod FILLERS_DA_RNN da_fillers rnn none bucket 1

