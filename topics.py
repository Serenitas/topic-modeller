#!/usr/bin/python

import re
import artm
import time
import glob
import os
import matplotlib.pyplot as plt
import pymystem3 as mystem

batch_vectorizer = artm.BatchVectorizer(data_path='lemmed.txt', data_format='vowpal_wabbit', target_folder='batches')

dictionary = batch_vectorizer.dictionary

topic_num = 6
tokens_num = 50

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

model_plsa.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
model_plsa.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
model_plsa.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
model_plsa.scores.add(artm.TopTokensScore(name='top_tokens_score'))
model_plsa.scores.add(artm.TopicKernelScore(name='topic_kernel_score', probability_mass_threshold=0.3))

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

def print_measures(model_plsa, model_artm, model_lda):
    print ('Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['sparsity_phi_score'].last_value,
        model_artm.score_tracker['sparsity_phi_score'].last_value,
        model_lda.sparsity_phi_last_value))

    print ('Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['sparsity_theta_score'].last_value,
        model_artm.score_tracker['sparsity_theta_score'].last_value,
        model_lda.sparsity_theta_last_value))

    print ('Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_contrast,
        model_artm.score_tracker['topic_kernel_score'].last_average_contrast))

    print ('Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_purity,
        model_artm.score_tracker['topic_kernel_score'].last_average_purity))

    print('Kernel size: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_size,
        model_artm.score_tracker['topic_kernel_score'].last_average_size))

    print('Coherence: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_coherence,
        model_artm.score_tracker['topic_kernel_score'].last_average_coherence))

    print ('Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['perplexity_score'].last_value,
        model_artm.score_tracker['perplexity_score'].last_value,
        model_lda.perplexity_last_value))

    first_score = 0

    plt.plot(range(first_score, model_plsa.num_phi_updates),
             model_plsa.score_tracker['perplexity_score'].value[first_score:], 'b--',
             range(first_score, model_artm.num_phi_updates),
             model_artm.score_tracker['perplexity_score'].value[first_score:], 'r--',
             range(first_score, len(model_lda.perplexity_value)),
             model_lda.perplexity_value[first_score:], 'g--', linewidth=1)
    plt.xlabel('Число итераций')
    plt.ylabel('Перплексия')
    plt.grid(True)
    plt.show()


#print_measures(model_plsa, model_artm, model_lda)

print(model_artm.score_tracker['top_tokens_score'].last_tokens)

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


print_docs_by_topics(topic_num, topic_names, theta, 'topics_by_docs.txt')

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
for topic in all_tokens.keys():
    tokens = all_tokens[topic]
    ready_tokens[topic] = tokens[:10]

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
            ready_tokens[topic].append(ngram)
i = 0

for topic in ready_tokens.keys():
    topicfile.write(topic)
    topicfile.write(str(ready_tokens[topic]))
    topicfile.write('\n')

lemmed = open('lemmed.txt', mode='r', encoding='utf-8').readlines()



for filename in filenames:
    keywords = []
    filename = filename.strip('\n')
    print(filename + ": ")
    print(docs[i])
    words = lemmed[i].strip('\n').split(' ')
    for topic in docs[i]:
        topic = topic.split('|')
        print(topic[0] + ':')
        top_tokens = ready_tokens[topic[0]]
        for tok in top_tokens:
            if tok not in keywords:
                if  ' ' not in tok:
                   if tok in words:
                       print(tok)
                       keywords.append(tok)
                else:
                    allin = True
                    for w in tok.split(' '):
                        if w == '':
                            continue
                        if lemmer.lemmatize(w)[0] not in words:
                            allin = False
                            break
                    if allin:
                        print(tok)
                        keywords.append(tok)
    i = i + 1
















# model_artm.regularizers['sparse_phi_regularizer'].tau = 0.01
# model_artm.regularizers['sparse_theta_regularizer'].tau = -0.65
# model_artm.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

# model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
# 
# model_artm.save("model.txt", "model")
# for topic_name in model_artm.topic_names:
#     print (topic_name + ': ')
#     print (model_artm.score_tracker['top_tokens_score'].last_tokens[topic_name])

# print (model.score_tracker['perplexity_score'].value)
# print (model.score_tracker['sparsity_phi_score'].value)
# print (model.score_tracker['sparsity_theta_score'].value)
# print (model.score_tracker['top_tokens_score'].last_tokens)

# model.num_document_passes = 10
# model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

# print ("R1", model.score_tracker['perplexity_score'].value)

# model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=100)

# print ("R2", model.score_tracker['perplexity_score'].value)

# t1 = time.time()
# #model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=100)
#
# print ("ARTM", model.score_tracker['perplexity_score'].last_value,  time.time() - t1)
# print (model.score_tracker['sparsity_phi_score'].last_value)
# print (model.score_tracker['sparsity_theta_score'].last_value)
# print (model.score_tracker['top_tokens_score'].last_tokens)
#
# t1 = time.time()
# modelLda = artm.LDA(num_topics=20, dictionary=dictionary)
# modelLda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=100)
#
# print("LDA", modelLda.perplexity_last_value, time.time() - t1)
# print(modelLda.sparsity_phi_last_value)
# print(modelLda.sparsity_theta_last_value)
# print(modelLda.get_top_tokens())

def calc_coeffs():
    best_tau_phi = -5.0
    best_tau_theta = -5.0
    best_perplexity = 1000000

    print("Started parameters choosing")

    for i in range(-20, 20, 5):
        for j in range(-20, 20, 5):
            model_artm.regularizers['sparse_phi_regularizer'].tau = (i / 10.0)
            model_artm.regularizers['sparse_theta_regularizer'].tau = (j / 10.0)
            model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
            if model_artm.score_tracker['perplexity_score'].last_value < best_perplexity:
                best_perplexity = model_artm.score_tracker['perplexity_score'].last_value
                best_tau_phi = (i / 10.0)
                best_tau_theta = (j / 10.0)
                print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    print("RESULT 1 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    for i in range(int(10 * best_tau_phi) - 5, int(10 * best_tau_phi) + 5, 1):
        for j in range(int(10 * best_tau_theta) - 5, int(10 * best_tau_theta) + 5, 1):
            model_artm.regularizers['sparse_phi_regularizer'].tau = (i / 10.0)
            model_artm.regularizers['sparse_theta_regularizer'].tau = (j / 10.0)
            model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
            if model_artm.score_tracker['perplexity_score'].last_value < best_perplexity:
                best_perplexity = model_artm.score_tracker['perplexity_score'].last_value
                best_tau_phi = (i / 10.0)
                best_tau_theta = (j / 10.0)
                print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    print("RESULT 2 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    for i in range(int(100 * best_tau_phi) - 10, int(100 * best_tau_phi) + 10, 1):
        for j in range(int(100 * best_tau_theta) - 10, int(100 * best_tau_theta) + 10, 1):
            model_artm.regularizers['sparse_phi_regularizer'].tau = (i / 100.0)
            model_artm.regularizers['sparse_theta_regularizer'].tau = (j / 100.0)
            model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
            if model_artm.score_tracker['perplexity_score'].last_value < best_perplexity:
                best_perplexity = model_artm.score_tracker['perplexity_score'].last_value
                best_tau_phi = (i / 100.0)
                best_tau_theta = (j / 100.0)
                print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    print("RESULT 3 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)