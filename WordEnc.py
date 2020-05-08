from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


"""
#this section is for converting the pretrained-glove file to word2vec
glove_input_file = 'glove.6B/glove.6B.100d.txt'
word2vec_output_file = 'glove.6B/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
"""


class WordEncoding:

    def __init__(self):
    # load the Stanford GloVe model
        self.filename = 'glove.6B/glove.6B.100d.txt.word2vec'
        self.model = KeyedVectors.load_word2vec_format(self.filename, binary=False)

    def get_word_Vector(self, word):
        self.result = self.model.word_vec(word)
        return self.result


#initialize / load encoding data
WE = WordEncoding()

#example for getting back a vector
thisWord = WE.get_word_Vector('woman')
print(thisWord)
