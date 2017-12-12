

embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 20 -k 1 -r 0.0 -b 1 --t bucket --models rcnn none"
et="word2vec"
ed="600"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"


ds="pucrs_usp_pont"
myid="PONT_PUCRS_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save --save-predictions $myid $fixedparams


# LOAD METHOD
# fixedparams="--gpu -w 7 -e 20 -k 0 -r 0.0 -b 1 --t bucket --models rcnn none --load"
# time sudo python3 -m deepbond --id $myid -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams

