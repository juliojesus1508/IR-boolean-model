from context import *

files = filename
stops = list_stopwords

class PreProcessor:
    def __init__(self, stops):
        self.stop_words = stops
     
    def read(self, input_file):
        f = open("../data/documents/"+input_file, 'r')
        data = f.read()
        data = data.lower()
        f.close()
        return data

    def save(self, words, ouput_file):
        f = open('../data/preprocessed-data/'+ouput_file, 'w+')
        for word in words:
            f.write(word)
            f.write('\n')
        f.close()

    def clean_word(self, word):
        word = "".join([character for character in word if character not in string.punctuation]) 
        word = "".join([character for character in word if character not in string.digits]) 
        return word

    def clean_document(self, file):
        lemmatizer = WordNetLemmatizer()
        data = self.read(file)
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(data)
        words = [self.clean_word(word) for word in words]
        words = [word for word in words if word not in self.stop_words]
        words = [lemmatizer.lemmatize(word, pos="n")  for word in words]        
        words = np.unique(words)
        words = np.sort(words)        
        self.save(words,file)    
        return True

    def preprocessing(self, files):        
        for file in files:
            words = self.clean_document(file)                
        return True

if __name__ == '__main__':
    preprocess = PreProcessor(stops)
    PreProcessor.preprocessing(preprocess, files)
