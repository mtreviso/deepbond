# -*- coding: utf-8 -*-
'''
Created on 19/07/2014

@author: Roque Lopez
'''
from __future__ import unicode_literals
import os
import codecs
import pickle 
import re

class ProbabilisticTagger(object):
    '''
    Tagger based on the probability of ocurrence of a tag for a word
    '''

    def __init__(self, uppercase=False):
        self.__mac_morpho_list = self.__read_mac_morpho(uppercase)
        self.__delaf_list = self.__read_delaf()
    
    def tag(self, tokens):
        ''' Tag a list of tokens according to its POS tag most likely '''
        tmp_list = list()
        for token in tokens:
            token = token.strip()
            if token.lower() in self.__mac_morpho_list:
                tmp_list.append((token, self.__mac_morpho_list[token.lower()]))
            elif token.lower() in self.__delaf_list:
                tmp_list.append((token, self.__delaf_list[token.lower()]))
            else:
                tmp_list.append((token, 'N'))
        return tmp_list

    def vocabulary(self):
        s = set(self.__mac_morpho_list.values())
        return dict(zip(s, range(1, len(s)+1)))
    
    def __read_mac_morpho(self, uppercase):
        ''' Read the POS tags assigned in the corpus Mac-Morpho '''
        word_tag_list = dict()
        folder_path = "data/resource/lexical/mac_morpho"
        files = [file_text for file_text in os.listdir(folder_path) if file_text != 'README'] 
        
        for file_name in files:
            data_file = codecs.open(os.path.join(folder_path, file_name), 'r', encoding='latin1')
            lines = data_file.readlines()
            for line in lines:
                tmp_list = line.strip().split('_')
                if len(tmp_list) == 2:
                    word = tmp_list[0] if uppercase else tmp_list[0].lower()
                    tag = tmp_list[1].split('|')[0]
                    if not word in word_tag_list: word_tag_list[word] = dict()
                    if not tag in word_tag_list[word]: word_tag_list[word][tag] = 0
                    word_tag_list[word][tag] += 1
        
        for word, tags in word_tag_list.items():
            word_tag_list[word] = sorted(tags.items(), key=lambda x:x[1], reverse=True)[0][0]
        
        return word_tag_list
    
    def __read_delaf(self):
        ''' Read the POS tags assigned in the dictionary DELAF '''
        word_tag_list = dict()
        data_file = codecs.open("data/resource/lexical/DELAF_PB.dic", 'r', encoding='utf-16')
        lines = data_file.readlines()

        for line in lines:
            result = re.match('(.+),(.+)\.(\w+)(.*)', line.strip())
            word_tag_list[result.group(1)] = result.group(3)
            
        return word_tag_list
    
# if __name__ == '__main__':
# file_name = "data/resource/lexical/probabilistic_tagger.pkl"
# p_tagger = ProbabilisticTagger(uppercase=False)
# with open(file_name, 'wb') as handle:
#     pickle.dump(p_tagger, handle)

# with open(file_name, 'rb') as handle:
#     p_tagger = pickle.load(handle)

# print(p_tagger.tag(['a', 'cinderela', 'e', 'legal']))
#     