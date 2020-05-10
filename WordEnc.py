from torchnlp.word_to_vector import GloVe




class WordEncoding_auto:
    def __init__(self, dataset_type, vectorDim):
        self.vectors = GloVe(name=dataset_type, dim=vectorDim)
    
    def get_word_Vector(self, word):
        return self.vectors[word]




#"""
#example
WE = WordEncoding_auto('6B', 100)    # dataset types : ‘840B’, ‘twitter.27B’, ‘6B’, ‘42B’  || dimension: usually 100
thisWord = WE.get_word_Vector('dude')
print(thisWord)
# """