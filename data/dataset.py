import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from collections import Counter

class TextData(Dataset):

    def __init__(self, context_wordset: List[Tuple[int, List[int]]]):
        self.context_wordset = context_wordset
    
    def __getitem__(self, context_wordset_index) -> Tuple[torch.Tensor, int]:

        target, context = self.context_wordset[context_wordset_index]
        context_tensor = torch.tensor(context)

        return context_tensor, target
    
    def __len__(self) -> int:
        
        return len(self.context_wordset)
    
def preprocess_corpus(corpus: List[List[str]], min_freq: int, stopwords: List[str]) -> Tuple[List[List[str]], Dict[str, int], Dict[int, str]]:

    temp_counter = Counter([token.lower() for sentence in corpus for token in sentence])
    processed_corpus = [[token.lower() for token in sentence if token.isalpha() and temp_counter[token] > min_freq and token.lower() not in stopwords] for sentence in corpus]

    vocab_counter = Counter([token for sentence in processed_corpus for token in sentence])

    valid_words = [word for word, _ in vocab_counter.items()]
    word2id = {word:idx for idx, word in enumerate(valid_words)}
    id2word = {idx:word for idx, word in enumerate(valid_words)}

    return processed_corpus, word2id, id2word

def retrieve_context_words(preprocessed_corpus: List[List[str]], wid: Dict[str, int], context_size: int) -> List[tuple[int, List[int]]]:

    half_window = context_size // 2
    context_wordset = []

    for sentence in preprocessed_corpus:

        length = len(sentence)
        if length < context_size + 1:
            continue

        for idx, token in enumerate(sentence):

            target = wid[token]

            if idx < half_window:
                start = 0
                end = 0 + context_size + 1
            elif idx > length - half_window - 1:
                start = length - context_size - 1
                end = length
            else:
                start = idx - half_window
                end = idx + half_window + 1
            
            context = [wid[sentence[index]] for index in range(start, end) if index != idx]

            context_wordset.append((target, context))

    return context_wordset