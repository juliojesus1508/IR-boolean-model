from context import *

import string
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

#files = filename

class BooleanModel:
    def __init__(self):
        self.db_documents = dict()
        self.container = dict()
        self.logical_operators = ['and', 'or', 'not']
    
    #setup documents
    def set_db_documents(self):
        df = pd.read_csv('data/others/list_documents.txt', sep='\t', header=None, names=["any", "title", "author"])
        filename = df.iloc[:,0]
        title = df.iloc[:,1]
        author = df.iloc[:,2]
        for i in range(0,df.shape[0]):
            key = str(filename[i]) + '.txt'
            info = {
                "doc_id": i+1,
                "title": title[i],
                "author": author[i],
                "file": key
            }
            self.db_documents[key] = info
        #print('nro. de documentos: ', len(self.db_documents))
        return True

    #indexando
    def index_document(self, file):
        df = pd.read_csv('data/preprocessed-data/'+file, sep='\t', header=None, names=["filename"])
        words = df.iloc[:,0]
        for word in words:
            doc_id = self.db_documents[file]['doc_id']
            if word in self.container.keys():
                tmp = self.container[word]
                tmp.append(doc_id)                
                self.container[word] = tmp
            else:                                        
                tmp = [doc_id]
                self.container[word] = tmp
        
    def index_documents(self, files):
        for file in files:
            self.index_document(file)
        return True

    #parser
    def generate_keys(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        words = tokenizer.tokenize(sentence)   
        words = [lemmatizer.lemmatize(word, pos="n")  for word in words]      
        return words 


    #checking
    def cheking_word(self, sentence):             
        words = self.generate_keys(sentence)     
        for word in words:
            if word not in self.container.keys():
                if word not in self.logical_operators:
                    message = word + ' not found'
                    return message, False
        return 'checking successful', True


    #generando arrays boolean
    def generate_boolean_array(self, word):
        bool_array = np.zeros(len(self.db_documents), dtype=int)
        if word in self.container.keys():                           
            mask = self.container[word]
            mask = [ int(element) - 1 for element in mask]                
            bool_array[mask] = 1
        return bool_array

    def generate_boolean_arrays(self, words):
        bool_arrays = dict()
        for word in words:
            if word not in self.logical_operators:                             
                bool_arrays[word] = self.generate_boolean_array(word)
        return bool_arrays

    #operaciones booleanas
    def not_boolean_operation(self, boolean_array):
        binary_vec = np.logical_not( boolean_array )
        binary_vec = binary_vec.astype(int)        
        return binary_vec

    def solving_not_boolean_operation(self, words, boolean_arrays):
        tmp_boolean_arrays = boolean_arrays
        tmp_words = []
        idx = 0
        while idx < len(words):
            if words[idx] == 'not':
                binary_vec = self.not_boolean_operation( tmp_boolean_arrays[words[ idx+1 ]] )
                new_keyword = 'not ' + words[ idx+1 ]
                tmp_boolean_arrays[new_keyword] = binary_vec
                tmp_words.append(new_keyword)
                idx = idx + 2
            else:
                tmp_words.append(words[ idx ])
                idx = idx + 1
        return tmp_words, tmp_boolean_arrays

    def and_boolean_operation(self, bin_vec1, bin_vec2):
        return np.logical_and(bin_vec1, bin_vec2 )

    def or_boolean_operation(self, bin_vec1, bin_vec2):
        return np.logical_or(bin_vec1, bin_vec2 )

    def solving_query(self, sentence):
        words = self.generate_keys(sentence)
        bool_arrays = self.generate_boolean_arrays(words)
        words, bool_arrays = self.solving_not_boolean_operation(words, bool_arrays)
        idx = 1 
        seed = bool_arrays[words[0]] 
        while idx < len(words):
            if words[idx] == 'and':
                seed = self.and_boolean_operation(seed, bool_arrays[ words[idx+1] ] )
                idx = idx + 2
            elif words[idx] == 'or':
                seed = self.or_boolean_operation(seed, bool_arrays[ words[idx+1] ] )
                idx = idx + 2
            else:
                idx = idx + 1
        return seed

    #recuperando informacion de los documentos
    def information_retrieval(self, sentence):
        result = self.solving_query(sentence)
        ir = []
        idx = 0        
        for element in self.db_documents:
            if result[ idx ] == 1:
                ir.append(self.db_documents[element])
            idx = idx + 1
        return ir

        
    #exportar/importar index
    def export_container(self):        
        line = ''
        for k in self.container:
            line += str(k) + '\t'
            for e in self.container[k]:
                line += str(e) + '\t'
            line += '\n'
        f = open('data/load.txt', 'w+')
        f.write(line)        
        f.close()
        return True

    def import_container(self):
        with open('data/load.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:                
                line = line.strip().split('\t')
                key = line[0]
                values = []
                for value in range(1,len(line)):
                    values.append(line[value])
                self.container[key] = values
        return True
    


if __name__ == '__main__':
    #model = BoleanModel()
    #BoleanModel.set_db_docuemnts(model, files)
    #BoleanModel.get_db_documents(model)
    #BoleanModel.index_documents(model, files)
    #BoleanModel.export_container(model)
    #BoleanModel.import_container(model)
    print(':D')


