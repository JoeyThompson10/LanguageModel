import requests
from LanguageModel import NgramModel, Preprocessor

class TestNgramModel:
    def __init__(self, url):
        # Fetch and preprocess data from a URL
        response = requests.get(url)
        response.raise_for_status()  # Check for errors
        
        text_data = response.text
        self.words = Preprocessor.preprocess_text(text_data)
        self.model = NgramModel(n=3)  # Using a trigram model
        self.model.train(self.words)

    def test_predict_next_word(self):
        # Test the prediction function with different prefixes
        prefixes = ["better the", "your sample", "articles and"]
        
        for prefix in prefixes:
            prediction = self.model.predict_next_word(prefix)
            print(f"Given prefix '{prefix}', predicted next word is: {prediction}")
            # Add assertions to validate results if you have expected outcomes

    def test_generate_text(self):
        # Test the text generation with different seeds
        seeds = ["how tall is", "books are", "data processing"]
        
        for seed in seeds:
            generated_text = self.model.generate_text(seed)
            print(f"Given seed '{seed}', generated text is: {generated_text}")
            # Use assertions if you have expected outcomes to compare against

if __name__ == "__main__":
    # Sample URL from Project Gutenberg (e.g., "Pride and Prejudice" by Jane Austen)
    url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
    tester = TestNgramModel(url)
    tester.test_predict_next_word()
    tester.test_generate_text()
