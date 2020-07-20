# deepbond
Deep neural approach to Boundary and Disfluency Detection

This is part of my MSc project. More info:
\
[My dissertation (ptbr)](http://www.teses.usp.br/teses/disponiveis/55/55134/tde-05022018-090740/pt-br.php)
 • 
[EACL paper](http://www.aclweb.org/anthology/E17-1030)
 • 
[STIL paper](http://aclweb.org/anthology/W17-6618)
 • 
[PROPOR paper](https://www.researchgate.net/publication/327223308_Sentence_Segmentation_and_Disfluency_Detection_in_Narrative_Transcripts_from_Neuropsychological_Tests_13th_International_Conference_PROPOR_2018_Canela_Brazil_September_24-26_2018_Proceedings)
 • 
[LREC paper](https://www.aclweb.org/anthology/2020.lrec-1.317.pdf)


# Installation 

First, clone this repository using `git`:

```sh
git clone https://github.com/mtreviso/deepbond.git
```

 Then, `cd` to the DeepBond folder:
```sh
cd deepbond
```

Create a Python virtualenv and install all dependencies 
using:
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run the install command:
```sh
python3 setup.py install
```

Please note that since Python 3 is required, all the above commands (pip/python) 
have to be bounded to the Python 3 version.



# Data
The data should be put in a folder called `data` in the root dir. Here is the basic ingredients that you might need:

- Corpus (see license): https://github.com/Edresson/DNLT-BP
- Word embeddings (word2vec skipgram): https://www.dropbox.com/s/rw3ti4ebctufp4j/embeddings.zip?dl=1
- Prosodic information (only for Control and MCI): https://www.dropbox.com/s/0gmt2o2xeah13xk/prosodic.zip?dl=1  

You can also send me an e-mail if you have any questions!




# Usage
You can use deepbond in two ways:
* Via CLI interface ([example](https://github.com/mtreviso/deepbond/blob/master/experiments/train_ccl_rcnn_crf.sh))
* Import as a library ([example](https://github.com/mtreviso/deepbond/blob/master/experiments/example_training_cinderela.py))

The full list of arguments (CLI) and options (lib) can be seen via:
```sh
python3 -m deepbond --help
```

Take a look at the `experiments` folder for more examples.


# License
MIT.


# Cite

If you use deepbond, you can cite this paper:

```
@inproceedings{treviso2018sentence,
  author = "Marcos Vinícius Treviso and Sandra Maria Aluísio",
  title = "Sentence Segmentation and Disfluency Detection in Narrative Transcripts from Neuropsychological Tests",
  booktitle = "Computational Processing of the Portuguese Language (PROPOR)",
  year = "2018",
  publisher = "Springer International Publishing",
  pages = "409--418",
}
```

Or the more recent publication (results without prosodic information + CRF)
```
@inproceedings{casanova-etal-2020-evaluating,
    title = "Evaluating Sentence Segmentation in Different Datasets of Neuropsychological Language Tests in {B}razilian {P}ortuguese",
    author = {Casanova, Edresson  and
      Treviso, Marcos  and
      H{\"u}bner, Lilian  and
      Alu{\'\i}sio, Sandra},
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference (LREC)",
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    pages = "2605--2614",
    ISBN = "979-10-95546-34-4",
}
```




