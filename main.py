import logging
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import brown

from config.config import Config
from model.cbow import CBOW
from train.training import train_epoch, test_epoch, TrainInfo
from data.dataset import preprocess_corpus, retrieve_context_words, TextData

def main():

    logging.basicConfig(filename = "epoch", level = logging.INFO)
    logger = logging.getLogger(__name__)

    torch.set_default_device(Config.DEVICE)
    original_corpus = brown.sents()
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'is', 'of', 
                'that', 'the', 'was',  'were', 'will', 'very']

    preprocessed_corpus, wid, idw = preprocess_corpus(original_corpus, Config.MIN_FREQ, stopwords)
    context_wordset = retrieve_context_words(preprocessed_corpus, wid, Config.CONTEXT_SIZE)

    WORDSET_LENGTH = len(context_wordset)
    VOCAB_LENGTH = len(wid)
    DATASET = TextData(context_wordset)
    W2V = CBOW(VOCAB_LENGTH, Config.EMBEDDING_DIM)
    CRITERION = nn.CrossEntropyLoss()
    OPTIMIZER = optim.AdamW(W2V.parameters(), lr = Config.LEARNING_RATE)

    train_size = int(Config.TRAIN_SPLIT * WORDSET_LENGTH)
    test_size = WORDSET_LENGTH - train_size

    train_dataset, test_dataset = random_split(DATASET, [train_size, test_size], generator = Config.GENERATOR)

    train_dataloader = DataLoader(train_dataset, batch_size = Config.BATCH_SIZE, shuffle = True, generator = Config.GENERATOR)
    test_dataloader = DataLoader(test_dataset, batch_size = Config.BATCH_SIZE, shuffle = True, generator = Config.GENERATOR)

    for epoch in range(Config.NUM_EPOCHS):

        print(f"EPOCH: {epoch + 1}")

        train_loss = train_epoch(W2V, train_dataloader, CRITERION, OPTIMIZER)
        TrainInfo.train_losses.append(train_loss)

        test_loss = test_epoch(W2V, test_dataloader, CRITERION)
        TrainInfo.test_losses.append(test_loss)

        logger.info(f"EPOCH: {epoch + 1} || Training Loss = {train_loss} | Testing Loss = {test_loss}")

    torch.save(W2V.state_dict(), "saved/cbow_model.pth")
    torch.save(wid, "saved/wid.pth")

if __name__ == "__main__":
    main()