import re
from collections import defaultdict, Counter
import random

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)
        self.vocab = set()  # Set to store unique words from the training data

    def train(self, words):
        for i in range(len(words) - self.n + 1):
            prefix = tuple(words[i: i + self.n - 1])
            next_word = words[i + self.n - 1]
            self.model[prefix][next_word] += 1
            self.vocab.add(next_word)

    def predict_next_word(self, prefix):
        prefix = tuple(Preprocessor.preprocess_text(prefix))
        
        if len(prefix) != self.n - 1:
            raise ValueError(f"Expected a prefix of length {self.n-1} words, but received a prefix with {len(prefix)} words.")
        
        if prefix not in self.model:
            return random.choice(list(self.vocab))  # Return a random word if prefix is unknown
        
        # Sampling a word based on its probability
        words, counts = zip(*self.model[prefix].items())
        total = sum(counts)
        probabilities = [count/total for count in counts]
        return random.choices(words, probabilities)[0]

    def generate_text(self, seed, num_words=50):
        current_text = Preprocessor.preprocess_text(seed)
        
        if len(current_text) > self.n - 1:
            current_text = current_text[-(self.n - 1):]
        
        for _ in range(num_words):
            next_word = self.predict_next_word(' '.join(current_text[-(self.n-1):]))
            current_text.append(next_word)
        
        return ' '.join(current_text)

class Preprocessor:
    @staticmethod
    def preprocess_text(text):
        return re.findall(r'\w+', text.lower())
