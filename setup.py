from setuptools import setup
from setuptools import find_packages

setup(name='deepbond',
      version='1.0',
      description='Deep neural approach to Boundary and Disfluency Detection',
      author='Marcos Treviso',
      author_email='marcosvtreviso@gmail.com',
      install_requires=['numpy>=1.11.0',
                        'keras>=2.1.5',
                        'pandas>=0.17.1',
                        'scikit_learn>=0.19.1',
                        'nltk>=3.2.5',
                        'nlpnet>=1.2.2',
                        'gensim>=2.0.0'],
      extras_require={
          'error_analysis': ['pandas_ml', 'DocumentFeatureSelection'],
          'crfmodels': ['sklearn_crfsuite>=1.4.1']
      },
packages=find_packages())
