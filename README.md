# Lightweight-Neural-Models-for-BGP-Hijack-Detection

This project investigates lightweight neural architectures for detecting BGP prefix hijacking using AS-path data. Starting from an existing CNN-LSTM baseline with ASN embeddings, we explore whether simpler models such as CNN-only and CNN-GRU can achieve comparable detection performance with lower computational complexity.

This repository contains a TensorFlow/Keras-based classifier for detecting BGP hijacking events from AS path data. The model uses pre-trained BGP2Vec embeddings to represent AS numbers and supports three neural network architectures:

- CNN-LSTM
- CNN-GRU
- CNN-Only

The script reads labeled BGP paths, converts each AS path into a sequence of BGP2Vec embedding indices, trains a binary classifier, evaluates the model, and saves the trained model.

---

## File

lstm_hijack_classifier.py

## Usage

python3 lstm_hijack_classifier.py <bgp2vec_model> <labeled_paths_file> <output_model> <model_selection>

## Example

python3 lstm_hijack_classifier.py bgp2vec/2days_2020.b2v classified/2days_2020.vf lstm/2days_2020.keras 0
