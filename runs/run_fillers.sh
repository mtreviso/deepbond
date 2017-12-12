# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_fillers"

embpath="/media/treviso/Recovered Volume/EmbeddingsDeepbond/ptbr"
fixedparams="--gpu -w 7 -e 25 -k 5 -b 1 -t bucket --models rcnn rcnn"
et="word2vec"
ed="600"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
et="fonseca"
ef="data/embeddings/fonseca/"


ds="controle_fillers"
myid="FILLERS_CTR_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef --save --save-predictions $myid $fixedparams

ds="ccl_fillers"
myid="FILLERS_CCL_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef --save --save-predictions $myid $fixedparams

ds="da_fillers"
myid="FILLERS_DA_${ds}_${et}_${ed}_${ea}"
fixedparams="--gpu -w 7 -e 30 -k 5 -b 1 -t bucket --models rcnn none"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef --save --save-predictions $myid $fixedparams
