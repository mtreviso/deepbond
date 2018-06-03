from deepbond.task import SentenceBoundaryDetector, FillerDetector, EditDisfDetector
from deepbond import Pipeline

def print_predictions(texts):
	for t in texts:
		print(' '.join(list(map(lambda x: x[0]+x[1], t))))

texts = ['ela morava com a madrasta as irmã né e ela era diferenciada das três era maltratada ela tinha que fazer limpeza na casa toda no castelo alias e as irmãs não faziam nada',
		'aqui é uma menininha simples uhn humilde eu creio creio que era humilde tava com os pais vivia com os pais depois é ela tinha é esse cavalo esse animalzinho de estimação depois ela foi morar no palácio',
		'era uma vez uma uma menina uma garota né que vivia numa castelo com o pai e ela gostava muito de animais e ela estava ahn fazendo um passeio a cavalo e ela morava num castelo']


sbd = SentenceBoundaryDetector(l_model='rcnn', p_model='none', verbose=True)
sbd.set_model_id('SS_TEXT_CINDERELA')
preds = sbd.detect(texts=texts)
print_predictions(preds)

print('----')

fd = FillerDetector(l_model='rcnn', p_model='none', verbose=True)
fd.set_model_id('FILLERS_TEXT_CINDERELA')
preds = fd.detect(texts=texts)
print_predictions(preds)

print('----')

edd = EditDisfDetector(l_model='rcnn', p_model='none', verbose=True)
edd.set_model_id('EDITDISFS_TEXT_CINDERELA')
preds = edd.detect(texts=texts)
print_predictions(preds)
