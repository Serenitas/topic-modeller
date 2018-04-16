import artm
import matplotlib.pyplot as plt


def calc_coeffs():
    batch_vectorizer = artm.BatchVectorizer(data_path='lemmed.txt', data_format='vowpal_wabbit',
                                            target_folder='batches')

    dictionary = batch_vectorizer.dictionary

    topic_num = 10
    topic_names = ['topic_{}'.format(i) for i in range(topic_num)]
    model_artm = artm.ARTM(topic_names=topic_names, dictionary=dictionary, cache_theta=True)

    model_artm.scores.add(artm.PerplexityScore(name='perplexity_score', dictionary=dictionary))
    model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
    model_artm.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
    model_artm.scores.add(artm.TopTokensScore(name='top_tokens_score'))
    model_artm.scores.add(artm.TopicKernelScore(name='topic_kernel_score', probability_mass_threshold=0.3))

    model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='sparse_phi_regularizer'))
    model_artm.regularizers.add(artm.SmoothSparseThetaRegularizer(name='sparse_theta_regularizer'))
    model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='decorrelator_phi_regularizer'))

    best_tau_phi = -5.0
    best_tau_theta = -5.0
    best_perplexity = 1000000

    print("Started parameters choosing")

    for i in range(-20, 20, 5):
        for j in range(-20, 20, 5):
            model_artm.regularizers['sparse_phi_regularizer'].tau = (i / 10.0)
            model_artm.regularizers['sparse_theta_regularizer'].tau = (j / 10.0)
            model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=1)
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
            model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=100)
            if model_artm.score_tracker['perplexity_score'].last_value < best_perplexity:
                best_perplexity = model_artm.score_tracker['perplexity_score'].last_value
                best_tau_phi = (i / 100.0)
                best_tau_theta = (j / 100.0)
                print(best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

    print("RESULT 3 ", best_perplexity, " ", best_tau_phi, " ", best_tau_theta)

def print_measures(model_plsa, model_artm, model_lda):
    print('Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['sparsity_phi_score'].last_value,
        model_artm.score_tracker['sparsity_phi_score'].last_value,
        model_lda.sparsity_phi_last_value))

    print('Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['sparsity_theta_score'].last_value,
        model_artm.score_tracker['sparsity_theta_score'].last_value,
        model_lda.sparsity_theta_last_value))

    print('Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_contrast,
        model_artm.score_tracker['topic_kernel_score'].last_average_contrast))

    print('Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_purity,
        model_artm.score_tracker['topic_kernel_score'].last_average_purity))

    print('Kernel size: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_size,
        model_artm.score_tracker['topic_kernel_score'].last_average_size))

    print('Coherence: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['topic_kernel_score'].last_average_coherence,
        model_artm.score_tracker['topic_kernel_score'].last_average_coherence))

    print('Background tokens ratio: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['background_tokens_ratio_score'].last_value,
        model_artm.score_tracker['background_tokens_ratio_score'].last_value))

    #percent_plsa = (int(model_plsa.score_tracker['class_precision_score'].last_error) / int(
        #model_plsa.score_tracker['class_precision_score'].last_total)) * 100
    #percent_artm = (int(model_artm.score_tracker['class_precision_score'].last_error) / int(
        #model_artm.score_tracker['class_precision_score'].last_total)) * 100
    print('Class precision: {0:.3f} of {1:.3f} errors (PLSA) vs. {2:.3f} of {3:.3f} errors (ARTM)'.format(
        model_plsa.score_tracker['class_precision_score'].last_error,
        model_artm.score_tracker['class_precision_score'].last_error,
        model_plsa.score_tracker['class_precision_score'].last_total,
        model_artm.score_tracker['class_precision_score'].last_total))

    print('Topic mass phi - mass:')
    print(model_plsa.score_tracker['topic_mass_phi_score'].last_topic_mass)
    print(model_artm.score_tracker['topic_mass_phi_score'].last_topic_mass)

    print('Topic mass phi - ratio:')
    print(model_plsa.score_tracker['topic_mass_phi_score'].last_topic_ratio)
    print(model_artm.score_tracker['topic_mass_phi_score'].last_topic_ratio)

    print('Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM) vs. {2:.3f} (LDA)'.format(
        model_plsa.score_tracker['perplexity_score'].last_value,
        model_artm.score_tracker['perplexity_score'].last_value,
        model_lda.perplexity_last_value))

    first_score = 10

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

    plt.plot(range(first_score, model_plsa.num_phi_updates),
             model_plsa.score_tracker['topic_kernel_score'].average_contrast[first_score:], 'b--',
             range(first_score, model_artm.num_phi_updates),
             model_artm.score_tracker['topic_kernel_score'].average_contrast[first_score:], 'r--')
    plt.xlabel('Число итераций')
    plt.ylabel('average_contrast')
    plt.grid(True)
    plt.show()

    plt.plot(range(first_score, model_plsa.num_phi_updates),
             model_plsa.score_tracker['topic_kernel_score'].average_purity[first_score:], 'b--',
             range(first_score, model_artm.num_phi_updates),
             model_artm.score_tracker['topic_kernel_score'].average_purity[first_score:], 'r--')
    plt.xlabel('Число итераций')
    plt.ylabel('average_purity')
    plt.grid(True)
    plt.show()

    plt.plot(range(first_score, model_plsa.num_phi_updates),
             model_plsa.score_tracker['topic_kernel_score'].average_size[first_score:], 'b--',
             range(first_score, model_artm.num_phi_updates),
             model_artm.score_tracker['topic_kernel_score'].average_size[first_score:], 'r--')
    plt.xlabel('Число итераций')
    plt.ylabel('average_size')
    plt.grid(True)
    plt.show()


def experiment(filename):
    batch_vectorizer = artm.BatchVectorizer(data_path=filename, data_format='vowpal_wabbit',
                                            target_folder='batches')

    dictionary = batch_vectorizer.dictionary

    topic_num = 30
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

    passes = 100
    model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)
    model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)
    model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=passes)

    print_measures(model_plsa, model_artm, model_lda)
