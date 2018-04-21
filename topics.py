#!/usr/bin/python

import artm
import time
import pymystem3 as mystem
import tf_idf_builder

start = time.time()

batch_vectorizer = artm.BatchVectorizer(data_path='lemmed.txt', data_format='vowpal_wabbit', target_folder='batches')

dictionary = batch_vectorizer.dictionary

topic_num = 10
tokens_num = 100
print("ARTM training")
topic_names = ['topic_{}'.format(i) for i in range(topic_num)]
model_artm = artm.ARTM(topic_names=topic_names, dictionary=dictionary, cache_theta=True)
model_plsa = artm.ARTM(topic_names=topic_names, cache_theta=True,
                       scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)])
model_lda = artm.LDA(num_topics=topic_num)

model_artm.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
model_artm.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
model_artm.scores.add(artm.TopTokensScore(name='top_tokens_score', num_tokens=tokens_num))
model_artm.scores.add(artm.TopicKernelScore(name='topic_kernel_score', probability_mass_threshold=0.3))
model_artm.scores.add(artm.BackgroundTokensRatioScore(name='background_tokens_ratio_score'))
model_artm.scores.add(artm.ClassPrecisionScore(name='class_precision_score'))
model_artm.scores.add(artm.TopicMassPhiScore(name='topic_mass_phi_score'))

model_plsa.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
model_plsa.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
model_plsa.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
model_plsa.scores.add(artm.TopTokensScore(name='top_tokens_score'))
model_plsa.scores.add(artm.TopicKernelScore(name='topic_kernel_score', probability_mass_threshold=0.3))
model_plsa.scores.add(artm.BackgroundTokensRatioScore(name='background_tokens_ratio_score'))
model_plsa.scores.add(artm.ClassPrecisionScore(name='class_precision_score'))
model_plsa.scores.add(artm.TopicMassPhiScore(name='topic_mass_phi_score'))

model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
model_artm.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))

model_artm.regularizers['sparse_phi_regularizer'].tau = 0.01
model_artm.regularizers['sparse_theta_regularizer'].tau = -1.06
# model_artm.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

model_plsa.initialize(dictionary=dictionary)
model_artm.initialize(dictionary=dictionary)
model_lda.initialize(dictionary=dictionary)

passes = 10
model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)
model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)


# print(model_artm.score_tracker['top_tokens_score'].last_tokens)

theta = model_artm.get_theta(topic_names)
phi = model_artm.get_phi(topic_names)


def print_docs_by_topics(topic_num, topic_names, theta, filename):
    threshold = 1.0 / topic_num

    docs_by_topics_file = open(filename, mode='w', encoding='utf-8')

    for i in range(theta.count('columns').size):
        docs_by_topics_file.write(topic_names[i] + ": ")
        for j in range(theta.count('index').size):
            if theta[j][i] > threshold:
                docs_by_topics_file.write(str(j) + "|" + str(theta[j][i]) + ' ')
        docs_by_topics_file.write('\n')


# for each topic prints documents in which probability of this topic greater than threshold
print("Printing docs by topics")
print_docs_by_topics(topic_num, topic_names, theta, 'topics_by_docs.txt')

print("Computing keyphrases")
filenames = open('filenames.txt', mode='r', encoding='utf-8').readlines()
doc_topics = open('topics_by_docs.txt', mode='r', encoding='utf-8').readlines()

docs = []
for i in range(theta.count('index').size):
    docs.append([])

for topic in doc_topics:
    topic = topic.strip('\n').split(' ')
    topic_name = 'undefined'
    for doc in topic:
        if '|' not in doc:
            topic_name = doc.strip(':')
        else:
            index = doc.split('|')[0]
            prob = doc.split('|')[1]
            docs[int(index)].append(topic_name + '|' + prob)

all_tokens = model_artm.score_tracker['top_tokens_score'].last_tokens
ready_tokens = model_artm.score_tracker['top_tokens_score'].last_tokens
ngrams_tokens = model_artm.score_tracker['top_tokens_score'].last_tokens
for topic in ngrams_tokens:
    ngrams_tokens[topic] = []
#for topic in all_tokens.keys():
#    tokens = all_tokens[topic]
#    ready_tokens[topic] = tokens[:10]

ngrams = open('adapted.txt', mode='r', encoding='utf-8').read().split('\n')
lemmer = mystem.Mystem()
topicfile = open('topics.txt', mode='w', encoding='utf-8')
tokens = []

for ngram in ngrams:
    if ngram == '':
        continue
    for topic in all_tokens.keys():
        tokens = all_tokens[topic]
        all_in = True
        for word in ngram.strip('\n').split(' '):
            if lemmer.lemmatize(word)[0] not in tokens:
                all_in = False
                break
        if all_in:
            ngrams_tokens[topic].append(ngram)
i = 0

for topic in ready_tokens.keys():
    topicfile.write(topic)
    topicfile.write(str(ready_tokens[topic][:10] + ngrams_tokens[topic]))
    topicfile.write('\n')

lemmed = open('lemmed.txt', mode='r', encoding='utf-8').readlines()

tf_idf_dict = tf_idf_builder.build_td_idf_dict(lemmed)

docngrams = open(file='doc_ngrams.txt', mode='r', encoding='utf-8').read().split('\n')
ngrams_by_doc = dict()
for doc in docngrams:
    if ':' not in doc:
        continue
    doc = doc.split(':')
    filename = doc[0]
    ngrams = doc[1].split('|')
    ngrams_by_doc[filename] = ngrams


def is_in(tok, topwords):
    for tuple in topwords:
        if tuple[0] == tok:
            return True
    return False


for filename in filenames:
    keywords = []
    filename = filename.strip('\n')
    print(filename + ": ")
    print(docs[i])
    topwords = tf_idf_dict[i][:50]
    words = lemmed[i].strip('\n').split(' ')
    for topic in docs[i]:
        topic = topic.split('|')
        print(topic[0] + ':')
        prob = topic[1]
        size = int(tokens_num * float(prob))
        top_tokens = ready_tokens[topic[0]][:size] + ngrams_tokens[topic[0]]
        for tok in top_tokens:
            if len(tok) < 2:
                continue
            if ' ' not in tok:
                if is_in(tok, topwords[:10]):
                    print(tok)
            else:
                if tok in ngrams_by_doc[filename]:
                    all_in = True
                    for w in tok.split(' '):
                        if w == '':
                            continue
                        if is_in(lemmer.lemmatize(w)[0], topwords):
                            all_in = False
                            break
                    if all_in:
                        print(tok)
    i = i + 1

print("Time: ", time.time() - start)



