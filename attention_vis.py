import matplotlib.pyplot as plt
import torch
import numpy as np
from word_encoding import WordEncodingAuto

WE = WordEncodingAuto('6B', 100)
thisWord = WE.get_word_vector('dude').reshape(1, -1)


def visualize(in_sentences, in_wordweights, in_sentenceweights, sentenceLength = 6, listed_sentences = 3):
    print(in_sentences[0:sentenceLength], " with wordweights_sentence1: ", in_wordweights[0])
    print("This sentence's weight is: ", in_sentenceweights[0,0])
    print(in_sentences[sentenceLength:sentenceLength * 2], " with wordweights_sentence2: ", in_wordweights[1])
    print("This sentence's weight is: ", in_sentenceweights[0,1])
    print(in_sentences[sentenceLength*2: sentenceLength*3], " with wordweights_sentence3: ", in_wordweights[2])
    print("This sentence's weight is: ", in_sentenceweights[0,2])

    highSentences, highSentenceIndxs = torch.topk(in_sentenceweights, listed_sentences, 1)

    words_with_attention = list()
    #numb_Sentences = len(highSentences[0])

    for ind in highSentenceIndxs[0]:
        for w in range(sentenceLength):
#            print("words: ", in_sentences[(ind*sentenceLength)+w])
            words_with_attention.append(in_sentences[(ind*sentenceLength)+w])

    print(words_with_attention)

  #  print(highSentences, " are highest weighted sentences", "|| length is: ", len(highSentences[0]))
    # Do heatmap viz
    # Plot it out
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5)
    heatmap = ax.pcolor(in_wordweights,cmap='RdBu')
    plt.colorbar(heatmap)
    titlestring = "Predicted Class: " #+ idx2genre[predmax] + ", True Class: " + example["genre"]
    plt.title(titlestring)
    #plt.colorbar()

    # Format
    fig = plt.gcf()

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(in_wordweights.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Set the labels

    # note I could have used nba_sort.columns but made "labels" instead
#    ax.set_yticklabels(np.round(in_sentenceweights,3), minor=False)

    #adjust here the number of sentences that should be visualized
    ax.set_yticklabels(('%.5f' % highSentences[0,0].item(), '%.5f' % highSentences[0,1].item(), '%.5f' % highSentences[0,2].item()))

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick2line.set_visible = True
        t.tick2line.set_visible = True
    for t in ax.yaxis.get_major_ticks():
        t.tick2line.set_visible = False
        t.tick2line.set_visible = False
    print(words_with_attention, "|| listen sentences amount is : ", listed_sentences)
    for y in range(listed_sentences):#(sentenceLength):
        no_words = min(sentenceLength, len(words_with_attention[y]))#min(sentenceLength, len(words_with_attention[y]))#len(sentences[y]))
       # print("amount no_words", no_words)
     #   print("y ",y)
    #    print("no words ",no_words)
        for x in range(no_words):
            if y == 0:
                textInd = x
                currSent = 0
            if y == 1:
                textInd = x+sentenceLength
                currSent = 1
            if y == 2:
                textInd = x+sentenceLength+sentenceLength
                currSent = 2
            print("x is: ", x, " while y is:", y, " and highsentence is: ", words_with_attention[textInd])

            thisWWeights = in_wordweights[currSent]
            thisWordsW = '%.5f' % thisWWeights[x].item() #repr(thisWWeights[x].item())
     #       print(thisWordsW)
            #sth = str(thisWWeights[textInd].item())
            #print(sth)
            thisText = words_with_attention[textInd] + '\n' + thisWordsW
            plt.text(x + 0.5, y + 0.5, thisText, horizontalalignment='center',
                     verticalalignment='center',)
            """plt.text(x + 0.5, y + 0.5, sentences[0][len(sentences[1])],#.decode('utf-8'),
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
            """
    #plt.savefig('rock_example.pdf')
    plt.show()


"""example:"""
sentences = ["1_word1","1_word2","1_word3","1_word4","1_word5","1_word6","2_word1","2_word2","2_word3","2_word4","2_word5","2_word6","3_word1","3_word2","3_word3","3_word4","3_word5","3_word6","4_word1","4_word2","4_word3","4_word4","4_word5","4_word6","5_word1","5_word2","5_word3","5_word4","5_word5","5_word6","6_word1","6_word2","6_word3","6_word4","6_word5","6_word6"]

#sentences = ["1_word1","1_word2","1_word3","1_word4","1_word5","1_word6","2_word1","2_word2","2_word3","2_word4","2_word5","2_word6","3_word1","3_word2","3_word3","3_word4","3_word5","3_word6"]

wordweights = torch.randn([3,6])
sentenceweights = torch.randn([1,6])
#print(len(sentences))
visualize(sentences, wordweights, sentenceweights)