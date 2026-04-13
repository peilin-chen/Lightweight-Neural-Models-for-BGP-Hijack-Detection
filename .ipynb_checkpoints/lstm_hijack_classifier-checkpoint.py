#!/usr/bin/env python3
#./lstm_hijack_classifier.py bgp2vec/2days_2020.b2v classified/2days_2020.vf lstm/2days_2020.keras 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_confusion_matrix(labels, predictions):
    conf_matrix_abs = tf.math.confusion_matrix(labels=labels,
                                               predictions=predictions)
    total = [(sum(c)) for c in conf_matrix_abs]
    return list([conf_matrix_abs[0]/total[0], conf_matrix_abs[1]/total[1]])


def get_weight_matrix(b2v):
    return np.vstack((np.zeros(b2v.vector_size), b2v.wv.vectors))


def main(args):
    b2v = KeyedVectors.load(args.b2v)

    data_df = pd.read_csv(args.labeled_paths, header=None,
                          converters={0: lambda x: x.split()})

    Xunpad = [[b2v.wv.key_to_index[asn] + 1 for asn in path] for path in data_df[0]]
    X = pad_sequences(Xunpad, maxlen=13, padding="post", truncating="pre",
                  value=0)

    Ydf = list(data_df[1] == 'GREEN')
    Y = [0 if y else 1 for y in Ydf]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    embedding_vectors = get_weight_matrix(b2v)

    vocab_size, embedding_size = b2v.wv.vectors.shape

    #print("args.model_selection:")
    #print(args.model_selection)

    if args.model_selection == '0':
        print("Use CNN-LSTM structure for training.")
        model = keras.Sequential([
        	layers.Input(shape=(13,)),
        	layers.Embedding(
        		input_dim=vocab_size + 1,
        		name="BGP2Vec",
        		output_dim=embedding_size,
        		mask_zero=True,
        		trainable=False,
        		weights=[embedding_vectors]
        	),
        	layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        	layers.MaxPooling1D(pool_size=2, strides=2),
        	layers.LSTM(100),
        	layers.Dense(1, activation='sigmoid')
        ])
    elif args.model_selection == '1':
        print("Use CNN-GRU structure for training.")
        model = keras.Sequential([
        	layers.Input(shape=(13,)),
        	layers.Embedding(
        		input_dim=vocab_size + 1,
        		name="BGP2Vec",
        		output_dim=embedding_size,
        		mask_zero=True,
        		trainable=False,
        		weights=[embedding_vectors]
        	),
        	layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        	layers.MaxPooling1D(pool_size=2, strides=2),
        	layers.GRU(100),
        	layers.Dense(1, activation='sigmoid')
        ])
    elif args.model_selection == '2':
        print("Use CNN-Only structure for training.")
        model = keras.Sequential([
        	layers.Input(shape=(13,)),
        	layers.Embedding(
        		input_dim=vocab_size + 1,
        		name="BGP2Vec",
        		output_dim=embedding_size,
        		mask_zero=True,
        		trainable=False,
        		weights=[embedding_vectors]
        	),
        	layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        	layers.MaxPooling1D(pool_size=2, strides=2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
        	layers.Dense(1, activation='sigmoid')
        ])

    model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

    model.summary()

    model.fit(
        np.asarray(x_train), np.asarray(y_train),
        validation_data=(np.asarray(x_test), np.asarray(y_test)), batch_size=64, epochs=10
    )

    #preds = model.predict_classes(x_test)
    pred_probs = model.predict(np.asarray(x_test))
    preds = (pred_probs > 0.5).astype(int).reshape(-1)
    m = np.array(get_confusion_matrix(y_test, preds))
    print('Confusion matrix:')
    print(m)

    model.save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('b2v', help='path to the trained bgp2vec model')
    parser.add_argument(
        'labeled_paths',
        help='path to file containing labeled the paths for training and testing'
    )
    parser.add_argument('output', help='path where the LSTM model will be saved')
    parser.add_argument('model_selection', help='0 for CNN-LSTM, 1 for CNN-GRU, 2 for CNN-Only')
    args = parser.parse_args()

    main(args)
