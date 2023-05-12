import sys
import re
import os

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import string

import sentencepiece as spm

class Preprocessor():

    def __init__(self, language = 'english', instream = None):
        self.language = language
        if instream:
            self.corpus_orig = self.read_string(instream)
        else:
            self.corpus_orig = self.read_articles(sys.stdin)
        self.corpus = self.preprocess(self.corpus_orig)

    def preprocess(self, corpus_list):
        preproc_corpus_list = []
        question_words = {'how', 'what', 'which', 'when', 'where', 'who', 'why'}
        stopset = stopwords.words(self.language) + list(string.punctuation)
        stopset = list(set(stopset) - question_words)
        for corpus in corpus_list:
            corpus = corpus.lower()
            corpus = " ".join([ i for i in word_tokenize(corpus) if i not in stopset ])
            preproc_corpus_list.append(corpus)
        return preproc_corpus_list

    def preprocess_sp(self, corpus_list):
        return self.sentencepiece_apply(self.preprocess(corpus_list))

    def sentencepiece_train(self, corpus_list, vocab_size = 24000):
        with open("./sp_corpus.txt", 'w') as fp_out:
            for corpus in corpus_list:
                print(corpus, file=fp_out)
        spm.SentencePieceTrainer.Train(f"--input=sp_corpus.txt --model_prefix=sp_model --vocab_size={vocab_size} --max_sentence_length=1000 --character_coverage=1.0 --num_threads=4 --hard_vocab_limit=false")
        return None

    def sentencepiece_apply(self, corpus_list):
        sp = spm.SentencePieceProcessor()
        sp.Load("./sp_model.model")
        return [" ".join(sp.EncodeAsPieces(corpus)) for corpus in corpus_list]

    def read_articles(self, fp):
        return [line.strip() for line in fp if line.strip() != ""]

    def read_string(self, mystring):
        return [mystring]


def main():
    vocab_size = sys.argv[1] if len(sys.argv) > 1 else 24000
    print("Create Preprocessor")
    preprocessor = Preprocessor(language = 'english')
    print("Train spm")
    preprocessor.sentencepiece_train(preprocessor.corpus, vocab_size = vocab_size)

if __name__ == "__main__":
    main()
