#!/usr/bin/python
#4143


import pymystem3   # learn more: https://python.org/pypi/pymystem3
import gensim
import re
import artm
import time
import glob
import os
import matplotlib.pyplot as plt

batch_vectorizer = artm.BatchVectorizer(data_path='D:\Work\magi\magi\python\project\\vw.mmro.txt',
                                        data_format='vowpal_wabbit',
                                        collection_name="mmro",
                                        target_folder='vw.mmro')

dictionary = artm.Dictionary()
dictionary.gather(data_path='test_mmro_batches_2')
dictionary.load_text(dictionary_path='test_mmro_batches/dictionary.txt')

topic_names = ['topic_{}'.format(i) for i in range(20)]
model_artm = artm.ARTM(topic_names=topic_names, num_topics=20, dictionary=dictionary)
model_plsa = artm.ARTM(topic_names=topic_names, cache_theta=True, scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)])

model_artm.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
model_artm.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
model_artm.scores.add(artm.TopTokensScore(name='top_tokens_score'))
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
model_artm.regularizers['sparse_theta_regularizer'].tau = -0.28
model_artm.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

model_plsa.initialize(dictionary=dictionary)
model_artm.initialize(dictionary=dictionary)

model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)


def print_measures(model_plsa, model_artm):
    print ('Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['sparsity_phi_score'].last_value,
        model_artm.score_tracker['sparsity_phi_score'].last_value))

    print ('Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['sparsity_theta_score'].last_value,
        model_artm.score_tracker['sparsity_theta_score'].last_value))

    print ('Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_contrast,
        model_artm.score_tracker['topic_kernel_score'].last_average_contrast))

    print ('Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_purity,
        model_artm.score_tracker['topic_kernel_score'].last_average_purity))

    print ('Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['perplexity_score'].last_value,
        model_artm.score_tracker['perplexity_score'].last_value))

    plt.plot(range(model_plsa.num_phi_updates),
             model_plsa.score_tracker['perplexity_score'].value, 'b--',
             range(model_artm.num_phi_updates),
             model_artm.score_tracker['perplexity_score'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
    plt.grid(True)
    plt.show()

print_measures(model_plsa, model_artm)

# best_tau_phi = -5.0
# best_tau_theta = -5.0
# best_perplexity = 10000
#
# print("Started parameters choosing")
#
# for i in range(-20, 20, 5):
#     for j in range(-20, 20, 5):
#      model.regularizers['sparse_phi_regularizer'].tau = (i / 10.0)
#      model.regularizers['sparse_theta_regularizer'].tau = (j / 10.0)
#      model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
#      if model.score_tracker['perplexity_score'].last_value < best_perplexity:
#          best_perplexity = model.score_tracker['perplexity_score'].last_value
#          best_tau_phi = (i / 10.0)
#          best_tau_theta = (j / 10.0)
#          print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)
#
# print("RESULT 1 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)
#
# for i in range(int(10 * best_tau_phi) - 5, int(10 * best_tau_phi) + 5, 1):
#     for j in range(int(10 * best_tau_theta) - 5, int(10 * best_tau_theta) + 5, 1):
#         model.regularizers['sparse_phi_regularizer'].tau = (i / 10.0)
#         model.regularizers['sparse_theta_regularizer'].tau = (j / 10.0)
#         model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
#         if model.score_tracker['perplexity_score'].last_value < best_perplexity:
#             best_perplexity = model.score_tracker['perplexity_score'].last_value
#             best_tau_phi = (i / 10.0)
#             best_tau_theta = (j / 10.0)
#             print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)
#
# print("RESULT 2 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)
#
# for i in range(int(100 * best_tau_phi) - 10, int(100 * best_tau_phi) + 10, 1):
#     for j in range(int(100 * best_tau_theta) - 10, int(100 * best_tau_theta) + 10, 1):
#         model.regularizers['sparse_phi_regularizer'].tau = (i / 100.0)
#         model.regularizers['sparse_theta_regularizer'].tau = (j / 100.0)
#         model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
#         if model.score_tracker['perplexity_score'].last_value < best_perplexity:
#             best_perplexity = model.score_tracker['perplexity_score'].last_value
#             best_tau_phi = (i / 100.0)
#             best_tau_theta = (j / 100.0)
#             print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)
#
# print("RESULT 3 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

model_artm.regularizers['sparse_phi_regularizer'].tau = 0.01
model_artm.regularizers['sparse_theta_regularizer'].tau = -0.28
model_artm.regularizers['decorrelator_phi_regularizer'].tau = 1e+5

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

model_artm.save("model.txt", "model")
for topic_name in model_artm.topic_names:
    print (topic_name + ': ')
    print (model_artm.score_tracker['top_tokens_score'].last_tokens[topic_name])

#print (model.score_tracker['perplexity_score'].value)
#print (model.score_tracker['sparsity_phi_score'].value)
#print (model.score_tracker['sparsity_theta_score'].value)
#print (model.score_tracker['top_tokens_score'].last_tokens)

#model.num_document_passes = 10
#model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

#print ("R1", model.score_tracker['perplexity_score'].value)

#model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=100)

#print ("R2", model.score_tracker['perplexity_score'].value)

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



