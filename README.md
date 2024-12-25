# Word2Vec: CBOW Model

This project implements a **Word2Vec** model using the **Continuous Bag of Words (CBOW)** architecture to learn word embeddings from a text corpus. The embeddings are visualized through dimensionality reduction using **t-SNE**.

---

## 🌟 Project Highlights
- **Word Embeddings**: Learns vector representations of words that capture their semantic relationships.
- **Dimensionality Reduction**: Embedding vectors are reduced to 2D for visualization, aiding in the understanding of word similarity and clustering.
- **Contextual Word Modeling**: The model predicts the target word using context words, learning associations between words that appear in similar contexts.

---

## 🏗️ Model Architecture
- **Embedding Layer**: Learns word representations in a high-dimensional vector space.
- **Hidden Layers**: Two fully connected layers to process the context and predict the target word.
- **Output Layer**: A linear layer that outputs logits for the target word prediction.

---

## 📂 Project Structure
- **config/**: Contains hyperparameter configurations.
- **data/**: Includes data preprocessing methods.
- **saved/**: Stores binary files with the trained model state and word-id mappings.
- **train/**: Methods related to the training process.
- **epoch/**: Logs the training loss for each epoch.
- **main.py**: The training loop and model saving functionality.
- **plotting.ipynb**: Notebook for dimensionality reduction of word embeddings, outputs a map in **embeddings_plot.png**.

---

## 📌 Dependencies
- **torch**: For model training and evaluation using PyTorch.
- **nltk**: For corpus handling.
- **scikit-learn**: Used for dimensionality reduction (t-SNE and PCA).
- **matplotlib**: For visualizing the word embeddings.

---
