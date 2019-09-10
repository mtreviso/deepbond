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

If you want to use my data, please send me an e-mail. A step-by-step tutorial (in portuguese) can be [accessed here](https://mtreviso.github.io/deepbond/tutorial.html).


# Installation 

First, clone this repository using `git`:

```sh
git clone https://github.com/mtreviso/deepbond.git
```

 Then, `cd` to the DeepBond folder:
```sh
cd deepbond
```

Automatically create a Python virtualenv and install all dependencies 
using `pipenv install`. And then activate the virtualenv with `pipenv shell`:
```sh
pip install --user pipenv
pipenv install
pipenv shell
```

Run the install command:
```sh
python setup.py install
```

Please note that since Python 3 is required, all the above commands (pip/python) 
have to bounded to the Python 3 version.


# Cite

If you use DeepBonDD, please cite this paper:

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

# License
MIT. See the [LICENSE](LICENSE) file for more details.


