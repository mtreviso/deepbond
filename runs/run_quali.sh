

embpath="/media/treviso/SAMSUNG/Embeddings/ptbr"
fixedparams="--gpu -w 7 -e 20 -k 5 -b 1 --t bucket"
et="word2vec"
ed="600"
ea="sg"
ef="${embpath}/${et}/pt_${et}_${ea}_${ed}.emb"

# sem embeddings
# sem pos
# sem prosodia
# dataset='controle ccl'
# extras='--without-emb --without-pos '
# for ds in $dataset; do
#     for extra in $extras; do
#     	myid="QUALI_${extra}_${ds}_${et}_${ed}_${ea}"
# 		time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams $extra
#     done
# done

# ds="controle"
# myid="QUALI_NORMAL_${ds}_${et}_${ed}_${ea}"
# time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams

# ds="ccl"
# myid="QUALI_NORMAL_${ds}_${et}_${ed}_${ea}"
# time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams


# dataset='constituicao constituicao_mini'
# extras='--without-emb --without-pos '
# for ds in $dataset; do
#     for extra in $extras; do
#     	myid="QUALI_${extra}_${ds}_${et}_${ed}_${ea}"
# 		time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams $extra
#     done
# done

ds="constituicao"
myid="QUALI_FINAL_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams

ds="constituicao_mini"
myid="QUALI_FINAL_${ds}_${et}_${ed}_${ea}"
time sudo python3 -m deepbond --id $myid --models rcnn rcnn -d $ds --emb-type $et --emb-file $ef --save-predictions $myid $fixedparams
