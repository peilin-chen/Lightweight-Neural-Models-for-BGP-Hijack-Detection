# Lightweight-Neural-Models-for-BGP-Hijack-Detection

This project investigates lightweight neural architectures for detecting BGP prefix hijacking using AS-path data. Starting from an existing CNN-LSTM baseline with ASN embeddings, we explore whether simpler models such as CNN-only and CNN-GRU can achieve comparable detection performance with lower computational complexity.

This repository contains a TensorFlow/Keras-based classifier for detecting BGP hijacking events from AS path data. The model uses pre-trained BGP2Vec embeddings to represent AS numbers and supports three neural network architectures:

- CNN-LSTM
- CNN-GRU
- CNN-Only

The script reads labeled BGP paths, converts each AS path into a sequence of BGP2Vec embedding indices, trains a binary classifier, evaluates the model, and saves the trained model.

---

## File

```bash
lstm_hijack_classifier.py
