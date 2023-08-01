import random
from array import array

import matplotlib
import nltk
import os
import seaborn as sns
import regex
#from keras.utils import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gensim.models import Word2Vec
import tensorflow as tf
# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# print(tf.config.list_physical_devices('GPU'))

##===========================ELMo_impots_pip========================================
#import tensorflow_hub as hub
import time
from datetime import datetime

import matplotlib
import nltk
import openpyxl
import os
import regex
import pandas as pd

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Bidirectional, TimeDistributed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from gensim.models import Word2Vec
from math import log
from nltk.tokenize import word_tokenize

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib import pyplot



#matplotlib.use('tkagg')

from nltk import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import numpy as np
from numpy import array
import tensorflow as tf

from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#===============================Nltk abstract_words tokenize======================================
with open('FGFSJournal.txt', 'rt', encoding='UTF8') as file:
    FGFS_abstract = []
    for line in file:
        if '<abstract>' in line:
            abstract = line.split('</abstract>')[0].split('<abstract>')[-1]
            abstract = ''.join(i for i in abstract if not i.isdigit())
            abstract = regex.sub('[^\w\d\s]+', '', abstract)
            ##abstract = nltk.sent_tokenize(abstract)
            abstract = nltk.word_tokenize(abstract)
            stop_words = set(stopwords.words('english'))
            filtered_sentence_abstract = [w.lower() for w in abstract if
                                          w.lower() not in punctuation and w.lower() not in stop_words]
            tagged_list = nltk.pos_tag(filtered_sentence_abstract)
            nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
            lm = WordNetLemmatizer()
            singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
            FGFS_abstract.append(singluar_form)



#print(FGFS_abstract)
#data = pd.DataFrame(data=FGFS_abstract)
# print(data)
#
# #
# # random.shuffle(FGFS_abstract)
# # train_data = FGFS_abstract[:4527]
# # test_data = FGFS_abstract[4527:]
# # print("train data:", len(train_data))
# # print("test data:", len(test_data))
#
#
# ##===============================pre-traind word2vec data==========================================
# CBOW_embeddings = Word2Vec(sentences=FGFS_abstract, vector_size=100, window=5, min_count=0, sg=0)
# CBOW_embeddings.wv.save_word2vec_format('CBOW_Pre-trained_word2Vec.txt', binary=False)
#
#
#
# ##******************abstract CNN training***********************************************
# print("create the tokenizer")
# token = Tokenizer()  # create the tokenizer
# token.fit_on_texts(FGFS_abstract)  # fit the tokenizer on the documents
# #print("Total words:", len(token.word_index))
#
#
# word_index = token.word_index
# #print('unique words: {}'.format(len(word_index)))
#
#
# # # print()
# vocab_size = len(token.word_index) + 1  # define vocabulary size (largest integer value)
# #print('Vocabulary size: %d' % vocab_size)
#
#
# #max_length = 259
# max_length = max(len(l) for l in FGFS_abstract) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
# #print('샘플의 최대 길이 : {}'.format(max_length))
#
#
# ##=======================================train_valid data split====================================
# train, test = train_test_split(FGFS_abstract, test_size=0.30, random_state=1000)
#
# print("train", len(train))
# print("valid", len(test))
#
#
# ##====================================train_labels====================================
# select_words = ['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
# train_labels = []
# for i in range(0, 3961):
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in FGFS_abstract[i]:
#             count += 1
#     if count >=1:
#         train_labels.append(1)
#     else:
#         train_labels.append(0)
#
#
# # ###====================================testation labels====================================
# # select_words =['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
# # valid_labels = []
# # for i in range(0, 1359):
# #
# #     count = 0
# #     for j in range(0, len(select_words)):
# #         if select_words[j] in FGFS_abstract[i]:
# #             count += 1
# #     if count >=1:
# #         valid_labels.append(1)
# #     else:
# #         valid_labels.append(0)
#
# ###====================================testation labels====================================
# select_words =['network', 'cloud', 'service', 'system', 'security', 'management', 'analysis', 'performance', 'model', 'resource']
# test_labels = []
# for i in range(0, 1698):
#
#     count = 0
#     for j in range(0, len(select_words)):
#         if select_words[j] in test[i]:
#             count += 1
#     if count >=2:
#         test_labels.append(1)
#     else:
#         test_labels.append(0)
#
#
# ### ======================train, valid and test data encoding===================================
# train_data = token.texts_to_sequences(train)
# #valid_data = token.texts_to_sequences(valid)
# test_data = token.texts_to_sequences(test)
#
#
# X_train = pad_sequences(train_data, maxlen=max_length, padding='post')
# y_train = np.asarray(train_labels).astype('float32').reshape((-1, 1))
#
# #X_valid = pad_sequences(valid_data, max_length, padding='post')
# #y_valid = np.asarray(valid_labels).astype('float32').reshape((-1, 1))
#
# X_test = pad_sequences(test_data, max_length, padding='post')
# y_test = np.asarray(test_labels).astype('float32').reshape((-1, 1))
#
#
# ##======================CNNs model with word2vec=====================================
# embedding_index = {}
# list_v = []
# file = open('CBOW_Pre-trained_word2Vec.txt', 'rt', encoding='UTF8')
# line = file.readline()
# totalWords, numOfFeatures = line.split()
# print(totalWords, numOfFeatures)
# for line in file:
#     values = line.split()
#     list_v.append(values)
#     word = values[0]
#     coefs = array(values[1:], dtype='float64')
#     embedding_index[word] = coefs
# #file.close()
#
# # print('Found %s word vectors.' % len(embedding_index))
# # df_values = pd.DataFrame(list_v)
# # print(df_values, "\n")
#
#
# embedding_matrix1 = np.array([[0 for col in range(100)] for row in range(12970)])
#
# for word, i in token.word_index.items():
#     embedding_vector = embedding_index.get(word)
#     if embedding_vector is not None:
#         if( i == 100):
#             print(i,"번째 완료")
#         for j in range(0, 100):
#             embedding_matrix1[i][j] = embedding_vector[j]
#        # print(i,"번째 완료")
#
# #print("embedding_matrix1:", embedding_matrix1)
#
#
##=======================CBOW_CNNs model using LSTM ====================================================
epochs = 100
embedding_dim = 100
pooling = 2
dropout = 0.2
filters = 128
batch_sizes = 128
validation_splits = 0.33


#=======================CBOW_CNNs model using BiLSTM ====================================================
cbow_cnn_Bilstm_model = Sequential()
cbow_cnn_Bilstm_model.add(Embedding(vocab_size, 100, weights=[embedding_matrix1], input_length=max_length, trainable=True))
cbow_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
cbow_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
cbow_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
cbow_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
cbow_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
cbow_cnn_Bilstm_model.add(Dense(10, activation='relu'))
cbow_cnn_Bilstm_model.add(Dense(1, activation='sigmoid'))
cbow_cnn_Bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
print(cbow_cnn_Bilstm_model.summary())
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')
c_bilstm_start_time = time.time()
CBOW_CNN_BiLSTM = cbow_cnn_Bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_sizes,
                                            validation_split=validation_splits, verbose=1)#, callbacks=[callback])
c_bilstm_end_time = time.time()
total_time = c_bilstm_end_time - c_bilstm_start_time
min = total_time / 60
#print("cnn_Bilstm_time:", cnn_Bilstm_time)
print("cnn_Bilstm training took {:} (h:mm:ss)".format(min))
CBOW_CNN_BiLSTM_train = cbow_cnn_Bilstm_model.evaluate(X_train, y_train, verbose=1)
print(('CBOW_CNN_LSTM_train_Score: %f' % (CBOW_CNN_BiLSTM_train[1] * 100)))
CBOW_CNN_BiLSTM_FGFS_Test = cbow_cnn_Bilstm_model.evaluate(X_test, y_test, verbose=1)
print(('CBOW_CNN_LSTM_FGFS_Test Accuracy: %f' % (CBOW_CNN_BiLSTM_FGFS_Test[1]*100)))


##====================F_score CBOW_cnn_gru==========================================
# predict probabilities for test set
CBOW_CNN_BiLSTM_probs = cbow_cnn_Bilstm_model.predict(X_test, verbose=1)
# reduce to 1d array
CBOW_CNN_BiLSTM_probs = CBOW_CNN_BiLSTM_probs[:, 0]


# accuracy: (tp + tn) / (p + n)
CBOW_CNN_BiLSTM_accuracy = accuracy_score(y_test, np.round(abs(CBOW_CNN_BiLSTM_probs)))
print('CBOW_CNN_BiLSTM_accuracy: %f' % CBOW_CNN_BiLSTM_accuracy)
# precision tp / (tp + fp)
CBOW_CNN_BiLSTM_precision = precision_score(y_test, np.round(abs(CBOW_CNN_BiLSTM_probs)))
print('CBOW_CNN_BiLSTM_precision: %f' % CBOW_CNN_BiLSTM_precision)
# recall: tp / (tp + fn)
CBOW_CNN_BiLSTM_recall = recall_score(y_test, np.round(abs(CBOW_CNN_BiLSTM_probs)))
print('CBOW_CNN_BiLSTM_recall_recall: %f' % CBOW_CNN_BiLSTM_recall)
# f1: 2 tp / (2 tp + fp + fn)
CBOW_CNN_BiLSTM_f1 = f1_score(y_test, np.round(abs(CBOW_CNN_BiLSTM_probs)))
print('CBOW_CNN_BiLSTM_f1: %f' % CBOW_CNN_BiLSTM_f1)

#
# ##========================Sg_pre-traind word2vec data===============================================
# sg_embeddings = Word2Vec(sentences=FGFS_abstract, vector_size=100, window=5, min_count=0, sg=1)
# sg_embeddings.wv.save_word2vec_format('Sg_Pre-trained_word2Vec.txt', binary=False)
#
#
# ### ====================================Sg_CNNs model with word2vec====================================
# sg_embedding_index = {}
# list_v = []
# file = open('Sg_Pre-trained_word2Vec.txt', 'rt', encoding='UTF8')
# line = file.readline()
# totalWords, numOfFeatures = line.split()
# print(totalWords, numOfFeatures)
# for line in file:
#     values = line.split()
#     list_v.append(values)
#     word = values[0]
#     coefs = array(values[1:], dtype='float64')
#     sg_embedding_index[word] = coefs
#
#
# print('Found %s word vectors.' % len(sg_embedding_index))
# df_values = pd.DataFrame(list_v)
# print(df_values, "\n")
#
# sg_embedding_matrix1 = np.array([[0 for col in range(100)] for row in range(12970)])
# for word, i in token.word_index.items():
#     # try:
#     embedding_vector = sg_embedding_index.get(word)
#     if embedding_vector is not None:
#         if( i == 100):
#             print(i,"번째 완료")
#         for j in range(0, 100):
#            sg_embedding_matrix1[i][j] = embedding_vector[j]
#         #print(i,"번째 완료")
#
# #print("sg_embedding_matrix1:", sg_embedding_matrix1)
#
#
#
# #=======================CBOW_CNNs model using BiLSTM ====================================================
# Sg_cnn_Bilstm_model = Sequential()
# Sg_cnn_Bilstm_model.add(Embedding(vocab_size, 100, weights=[sg_embedding_matrix1], input_length=max_length, trainable=True))
# Sg_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=2, padding='same', activation='relu'))
# Sg_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
# Sg_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
# Sg_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=3, padding='same', activation='relu'))
# Sg_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
# Sg_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
# Sg_cnn_Bilstm_model.add(Conv1D(filters=filters, kernel_size=4, padding='same', activation='relu'))
# Sg_cnn_Bilstm_model.add(MaxPooling1D(pool_size=pooling))
# Sg_cnn_Bilstm_model.add(Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=dropout)))
# Sg_cnn_Bilstm_model.add(Dense(10, activation='relu'))
# Sg_cnn_Bilstm_model.add(Dense(1, activation='sigmoid'))
# Sg_cnn_Bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Accuracy'])
# print(Sg_cnn_Bilstm_model.summary())
# #callback = tf.keras.callbacks.EarlyStopping(monitor='val_Accuracy', min_delta=0, patience=0, mode='auto')
# sg_c_bilstm_start_time = time.time()
# Sg_CNN_BiLSTM = Sg_cnn_Bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
#                                         batch_size=batch_sizes, verbose=1, validation_split=validation_splits)#, callbacks=[callback])
# sg_c_bilstm_end_time = time.time()
# sg_cnn_bilstm_time = sg_c_bilstm_end_time - sg_c_bilstm_start_time
# min = sg_cnn_bilstm_time / 60
# print("sg_cnn_bilstm training took {:} (h:mm:ss)".format(min))
# print("sg_cnn_bilstm_time:", sg_cnn_bilstm_time)
# Sg_CNN_BiLSTM_train = Sg_cnn_Bilstm_model.evaluate(X_train, y_train, verbose=1)
# print(('Sg_CNN_BiLSTM_train: %f' % (Sg_CNN_BiLSTM_train[1] * 100)))
# Sg_CNN_BiLSTM_FGFS_Test = Sg_cnn_Bilstm_model.evaluate(X_test, y_test, verbose=1)
# print(('Sg_CNN_BiLSTM_FGFS_Test Accuracy: %f' % (Sg_CNN_BiLSTM_FGFS_Test[1]*100)))
#
#
# ##====================F_score CBOW_cnn_gru==========================================
# # predict probabilities for test set
# Sg_CNN_BiLSTM_probs = Sg_cnn_Bilstm_model.predict(X_test, verbose=1)
# # reduce to 1d array
# Sg_CNN_BiLSTM_probs = Sg_CNN_BiLSTM_probs[:, 0]
#
#
# # accuracy: (tp + tn) / (p + n)
# Sg_CNN_BiLSTM_accuracy = accuracy_score(y_test, np.round(abs(Sg_CNN_BiLSTM_probs)))
# print('Sg_CNN_BiLSTM_accuracy: %f' % Sg_CNN_BiLSTM_accuracy)
# # precision tp / (tp + fp)
# Sg_CNN_BiLSTM_precision = precision_score(y_test, np.round(abs(Sg_CNN_BiLSTM_probs)))
# print('Sg_CNN_BiLSTM_precision: %f' % Sg_CNN_BiLSTM_precision)
# # recall: tp / (tp + fn)
# Sg_CNN_BiLSTM_recall = recall_score(y_test, np.round(abs(Sg_CNN_BiLSTM_probs)))
# print('Sg_CNN_BiLSTM_recall: %f' % Sg_CNN_BiLSTM_recall)
# # f1: 2 tp / (2 tp + fp + fn)
# Sg_CNN_BiLSTM_f1 = f1_score(y_test, np.round(abs(Sg_CNN_BiLSTM_probs)))
# print('Sg_CNN_BiLSTM_f1: %f' % Sg_CNN_BiLSTM_f1)
#
##================================CBOW model Acc figure========================================
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(CBOW_CNN_BiLSTM.history['Accuracy'], 'b', label="Training")
plt.plot(CBOW_CNN_BiLSTM.history['val_Accuracy'], 'g', label="Validation")

# Label the plot.
plt.title("CBOW Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('CBOW_Emb Model Acc')
plt.show()

# #
# # ##=================================Sg Model Acc Figure==========================================
# # # Use plot styling from seaborn.
# # sns.set(style='darkgrid')
# #
# # # Increase the plot size and font size.
# # sns.set(font_scale=1.5)
# # plt.rcParams["figure.figsize"] = (12, 6)
# #
# # # Plot the learning curve.
# # plt.plot(Sg_CNN_BiLSTM.history['Accuracy'], 'b', label="Training")
# # plt.plot(Sg_CNN_BiLSTM.history['val_Accuracy'], 'g', label="Validation")
# #
# # # Label the plot.
# # plt.title("Sg Training & Validation Accuracy")
# # plt.xlabel("Epoch")
# # plt.ylabel("Accuracy")
# # plt.legend()
# # plt.savefig('Sg_Emb Model Acc')
# # plt.show()
# #
# # ##================================CBOW model Loss figure========================================
# # # Use plot styling from seaborn.
# # sns.set(style='darkgrid')
# #
# # # Increase the plot size and font size.
# # sns.set(font_scale=1.5)
# # plt.rcParams["figure.figsize"] = (12, 6)
# #
# # # Plot the learning curve.
# # plt.plot(CBOW_CNN_BiLSTM.history['loss'], 'b', label="Training")
# # plt.plot(CBOW_CNN_BiLSTM.history['val_loss'], 'g', label="Validation")
# #
# # # Label the plot.
# # plt.title("CBOW Training & Validation Loss")
# # plt.xlabel("Epoch")
# # plt.ylabel("Loss")
# # plt.ylim(0, 5)
# # plt.legend()
# # plt.savefig('CBOW_Emb Model')
# # plt.show()
#
# ##=================================Sg Model Loss Figure==========================================
# # Use plot styling from seaborn.
# sns.set(style='darkgrid')
#
# # Increase the plot size and font size.
# sns.set(font_scale=1.5)
# plt.rcParams["figure.figsize"] = (12, 6)
#
# # Plot the learning curve.
# plt.plot(Sg_CNN_BiLSTM.history['loss'], 'b', label="Training")
# plt.plot(Sg_CNN_BiLSTM.history['val_loss'], 'g', label="Validation")
#
# # Label the plot.
# plt.title("Sg Training & Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim(0, 5)
# plt.legend()
# plt.savefig('Sg_Emb Model')
# plt.show()
