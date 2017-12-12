# ss/dd_fillers/dd_editdisfs/ssdd
task="dd_fillers"

embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 20 -k 0 -b 1 -t bucket --models rcnn none"
# et="word2vec"
# ed="600"
# ea="sg"
# ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"
et="fonseca"
ef="data/embeddings/fonseca/"

saveorload="--load"


ds="controle_fillers"
myid="FILLERS_CTR_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef $saveorload --save-predictions $myid $fixedparams

ds="ccl_fillers"
myid="FILLERS_CCL_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef $saveorload --save-predictions $myid $fixedparams

ds="da_fillers"
myid="FILLERS_DA_${ds}_${et}_${ed}_${ea}"
fixedparams="--gpu -w 7 -e 30 -k 5 -b 1 -t bucket --models rcnn none"
time sudo python3 -m deepbond --id $myid --task $task -d $ds --emb-type $et --emb-file $ef $saveorload --save-predictions $myid $fixedparams
