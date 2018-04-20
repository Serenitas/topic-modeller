import corpora_builder, ngram_adapter, coefficient_calculator
from time import time

directory = 'it_texts'
multiword_only_file = 'multiword_only.txt'

out_file = 'corpora.txt'
out_dictionary = 'dict.txt'
out_lemmatized = 'lemmed.txt'
out_ngrams = 'ngrams.txt'
out_ngrams_by_doc = 'doc_ngrams.txt'

#start = time()

#print("Building corpora")
corpora_builder.build_corpora(directory)
#print("Adapting ngrams")
#ngram_adapter.adapt_ngrams(out_ngrams, out_dictionary, 'result.txt')
#print("Printing ngrams by doc")
#corpora_builder.print_ngrams_by_doc(out_ngrams_by_doc, multiword_only_file, out_lemmatized)
#print("Time:", int((time() - start) / 60), 'min', int((time() - start) % 60), 'sec')
#coeffs = coefficient_calculator.calc_coeffs()
coefficient_calculator.experiment('lemmed.txt', 0, -3)

#0, -2.6 0, -3