import corpora_builder, ngram_adapter
from time import time

directory = 'all_texts'
out_file = 'corpora.txt'
out_dictionary = 'dict.txt'
out_lemmatized = 'lemmed.txt'
out_ngrams = 'ngrams.txt'

start = time()
corpora_builder.build_corpora(directory)
ngram_adapter.adapt_ngrams(out_ngrams, out_dictionary, 'result.txt')
print("Time: " + str(time() - start))