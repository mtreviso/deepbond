embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
et="word2vec"
ed="600"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
fixedparams="--gpu -w 7 -e 20 -k 5 --emb-type ${et} --emb-file ${ef}"

invokemethod(){
	local myid=$1
	local ds=$2
	local ma=$3
	local mb=$4
	local st=$5
	local bs=$6
	time sudo python3 -m deepbond --id $myid --save-predictions $myid --models $ma $mb -d $ds -t $st -b $bs $fixedparams
}

# CRF
# invokemethod MET_CRF_CTR controle crf crf dicted 1
# invokemethod MET_CRF_CCL ccl crf crf dicted 1
# invokemethod MET_CRF_CONST constituicao crf crf dicted 1
# invokemethod MET_CRF_CONST_MINI constituicao_mini crf crf dicted 1

# MLP
# invokemethod MET_MLP_CTR controle mlp mlp window 32
# invokemethod MET_MLP_CCL ccl mlp mlp window 32
# invokemethod MET_MLP_CONST constituicao mlp mlp window 32
# invokemethod MET_MLP_CONST_MINI constituicao_mini mlp mlp window 32

# CNN
# invokemethod MET_CNN_CTR controle cnn cnn bucket 1
# invokemethod MET_CNN_CCL ccl cnn cnn bucket 1
# invokemethod MET_CNN_CONST constituicao cnn cnn bucket 1
# invokemethod MET_CNN_CONST_MINI constituicao_mini cnn cnn bucket 1

# RCNN
# invokemethod MET_RCNN_CTR controle rcnn rcnn bucket 1
# invokemethod MET_RCNN_CCL ccl rcnn rcnn bucket 1
# invokemethod MET_RCNN_CONST constituicao rcnn rcnn bucket 1
# invokemethod MET_RCNN_CONST_MINI constituicao_mini rcnn rcnn bucket 1

# RNN
# invokemethod MET_RNN_CTR controle rnn rnn bucket 1
# invokemethod MET_RNN_CCL ccl rnn rnn bucket 1
# invokemethod MET_RNN_CONST constituicao rnn rnn bucket 1
# invokemethod MET_RNN_CONST_MINI constituicao_mini rnn rnn bucket 1

