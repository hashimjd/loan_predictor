#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from plotnine import *



# Load Data and convert all non-zero losses to 0 (keeping zero loss unchanged)
full_data = pd.read_csv('./Data/train_v2.csv', low_memory=False, index_col='id').dropna()
full_data['loss'] = (~(full_data['loss'] == 0)).astype(int)

# Split data for equal samples with loss = 0 and loss = 1.
# Note: column -1, the last column, is loss (our Y)
default_data = full_data[full_data['loss'] == 0].iloc[:500, np.r_[-1, :100]]
non_default_data = full_data[full_data['loss'] == 1].iloc[:500, np.r_[-1, :100]]


def train_validate(X_training, Y_training, X_validating, Y_validating, model):
    model.fit(X_training, Y_training)
    validating_accuracy = (model.predict(X_validating) == Y_validating).mean()
    return validating_accuracy


sample_sizes = list(range(100, 1001, 100))
svm_testing_accuracy = []
knn_testing_accuracy = []

for sample_size in sample_sizes:
    ## Splitting Sample into Training, Validating and Testing ##

    each_sample_size = int(sample_size / 2)

    non_default_sample = default_data.iloc[:each_sample_size, :]
    default_sample = non_default_data.iloc[:each_sample_size, :]

    # Training Data (first 50% of sample)
    training_data = pd.concat([non_default_sample.iloc[:int(0.5 * each_sample_size)],
                               default_sample.iloc[:int(0.5 * each_sample_size)]])
    X_training, Y_training = training_data.iloc[:, 1:], training_data.iloc[:, 0]

    # Validating Data (next 25% of sample)
    validating_data = pd.concat([non_default_sample.iloc[int(0.5 * each_sample_size):int(0.75 * each_sample_size)],
                                 default_sample.iloc[int(0.5 * each_sample_size):int(0.75 * each_sample_size)]])
    X_validating, Y_validating = validating_data.iloc[:, 1:], validating_data.iloc[:, 0]

    # Testing Data (last 25% of sample)
    testing_data = pd.concat([non_default_sample.iloc[int(0.75 * each_sample_size):],
                              default_sample.iloc[int(0.75 * each_sample_size):]])
    X_testing, Y_testing = testing_data.iloc[:, 1:], testing_data.iloc[:, 0]

    ## Transforming data with PCA ##

    # Apply PCA to "Training Data" to create PCA model
    # From Graph, only 1 Principal Components could explain more than 95% of variance in X
    explained_variance_ratio = PCA(n_components=7).fit(X_training).explained_variance_ratio_

    print(ggplot(aes(x=range(1, len(explained_variance_ratio) + 1), y=explained_variance_ratio)) +
          geom_line() +
          xlim(1, len(explained_variance_ratio)) +
          ylim(0, 1) +
          ggtitle(f'PCA: Principal Components vs Explained Variance Ratio\n'
                  f'(Sample Size = {sample_size}) based on Training Data') +
          xlab('Principal Components') +
          ylab('Explained Variance Ratio'))

    # Plot 1st vs 2nd Principal Components
    two_components = PCA(n_components=2).fit_transform(X_training, Y_training)
    print(ggplot(aes(x=two_components[:, 0], y=two_components[:, 1], color='factor(Y_training)')) +
          geom_point() +
          ggtitle(f'PCA: 1st Principal Component vs 2nd Pricipal Component\n'
                  f'(Sample Size = {sample_size}) based on Training Data') +
          xlab('1st Principal Component') +
          ylab('2nd Principal Component'))

    # Plot 1st Principal Component Histogram
    one_components = PCA(n_components=1).fit_transform(X_training, Y_training)
    print(ggplot(aes(x=one_components[:, 0], fill='factor(Y_training)')) +
          geom_density(alpha=0.5) +
          ggtitle(f'PCA: 1st Principal Component Histogram\n'
                  f'(Sample Size = {sample_size}) based on Training Data') +
          xlab('1st Principal Component') +
          ylab('Count'))

    # Select n_components = 1 and fit model with X_training only
    pca_model = PCA(n_components=1).fit(X_training)

    # Transform data with PCA
    transformed_X_training = pca_model.transform(X_training)
    transformed_X_validating = pca_model.transform(X_validating)
    transformed_X_testing = pca_model.transform(X_testing)

    ## Running SVM with Linear Kernel ##
    C = range(1, 100)
    svm_validating_accuracy = [train_validate(transformed_X_training, Y_training,
                                              transformed_X_validating, Y_validating,
                                              LinearSVC(C=c, dual=False)) for c in C]

    # Get best C value and corresponding LinearSVC model
    best_C = C[svm_validating_accuracy.index(max(svm_validating_accuracy))]
    best_svm = LinearSVC(C=best_C, dual=False).fit(transformed_X_training, Y_training)

    # Plot C vs Accuracy
    print(ggplot(aes(x=C, y=svm_validating_accuracy)) +
          geom_line() +
          ylim(0, 1) +
          ggtitle(f'SVM, Linear Kernal: C vs Validating Accuracy\n'
                  f'(Sample Size = {sample_size}, Best C = {best_C})') +
          xlab('C') +
          ylab('Validating Accuracy'))

    svm_testing_accuracy.append((best_svm.predict(transformed_X_testing) == Y_testing).mean())

    ## Running KNN ##
    K = range(1, 15)
    knn_validating_accuracy = [train_validate(transformed_X_training, Y_training,
                                              transformed_X_validating, Y_validating,
                                              KNeighborsClassifier(n_neighbors=k)) for k in K]

    # Get best K value and corresponding KNN model
    best_K = K[knn_validating_accuracy.index(max(knn_validating_accuracy))]
    best_knn = KNeighborsClassifier(n_neighbors=best_K).fit(transformed_X_training, Y_training)

    # Plot K vs Accuracy
    print(ggplot(aes(x=K, y=knn_validating_accuracy)) +
          geom_line() +
          ylim(0, 1) +
          ggtitle(f'KNN: K vs Validating Accuracy\n'
                  f'(Sample Size = {sample_size}, Best K = {best_K})') +
          xlab('K') +
          ylab('Validating Accuracy'))


    knn_testing_accuracy.append((best_knn.predict(transformed_X_testing) == Y_testing).mean())


# Plot Accuracy vs Sample Size
# Note: the chance is 0.5 because half of data has loss = 0 (and another half has loss = 1)
print(ggplot(pd.concat([pd.DataFrame({'model': 'linear svm',
                                      'testing accuracy': svm_testing_accuracy,
                                      'sample size': sample_sizes}),
                        pd.DataFrame({'model': 'knn',
                                      'testing accuracy': knn_testing_accuracy,
                                      'sample size': sample_sizes})]),
             aes(x='sample size',
                 y='testing accuracy',
                 color='model'))  +
      geom_line() +
      ylim(0, 1) +
      ggtitle('Linear SVM and KNN Testing Accuracy for different Sample Size') +
      xlab('Sample Size') +
      ylab('Testing Accuracy'))

