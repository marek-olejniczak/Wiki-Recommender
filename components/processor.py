import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
import time
import requests
from bs4 import BeautifulSoup

class TextProcessor:
    """
    A class to process text from the articles corpus, including tokenization,
    stemming, and lemmatization.
    """

    def __init__(self, corpus_path, output_path, batch_size=1000):
        """
        Initializes the TextProcessor.

        :param corpus_path: Path to the corpus CSV file.
        :param output_path: Path to save the processed data CSV.
        :param batch_size: The number of rows to process in each batch.
        """
        self.corpus_path = corpus_path
        self.output_path = output_path
        self.batch_size = batch_size
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _download_nltk_data(self):
        """Downloads necessary NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading necessary NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("NLTK data downloaded.")

    def _process_text(self, text):
        """
        Performs tokenization, stemming, and lemmatization on a single text string.
        """
        if not isinstance(text, str):
            return [], [], []
        
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        return filtered_tokens, stemmed_tokens, lemmatized_tokens

    def process_corpus(self):
        """
        Reads the corpus in batches, processes the text, and saves the result.
        """
        if not os.path.exists(self.corpus_path):
            print(f"Error: Corpus file not found at {self.corpus_path}")
            return

        print(f"Starting processing of '{self.corpus_path}'...")
        start_time = time.time()
        
        header = True
        chunk_count = 0

        for chunk in pd.read_csv(self.corpus_path, chunksize=self.batch_size):
            chunk_count += 1
            print(f"Processing batch {chunk_count}...")
            processed_data = chunk['text'].apply(self._process_text)
            
            chunk['tokens'] = processed_data.apply(lambda x: x[0])
            chunk['stemmed'] = processed_data.apply(lambda x: x[1])
            chunk['lemmatized'] = processed_data.apply(lambda x: x[2])
            
            # Columns to save
            output_chunk = chunk[['url', 'title', 'tokens', 'stemmed', 'lemmatized', 'content_hash']]
            output_chunk.to_csv(self.output_path, mode='a', header=header, index=False)
            
            if header:
                header = False
        
        end_time = time.time()
        print(f"Processing complete. Results saved to '{self.output_path}'.")
        print(f"Total processing time: {end_time - start_time:.2f} seconds.")

    def _scrap_and_process_url(self, url):
        """Scrapes a single URL, extracts text, and processes it."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.title.string if soup.title else 'No Title Found'
            
            content_div = soup.find('div', id='mw-content-text')
            if content_div:
                for tag in content_div.find_all(['table', 'sup']):
                    tag.decompose()
                text = content_div.get_text(separator=' ', strip=True)
            else:
                text = soup.body.get_text(separator=' ', strip=True)

            if not text:
                print(f"No text content found for {url}")
                return None

            tokens, stemmed, lemmatized = self._process_text(text)
            
            return {
                'url': url,
                'title': title,
                'tokens': tokens,
                'stemmed': stemmed,
                'lemmatized': lemmatized
            }
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None

    def process_history(self, urls, output_filename='processed_history.csv'):
        """
        Scrapes a list of URLs, processes their text content, and saves the results.

        :param urls: A list of URLs to scrape and process.
        :param output_filename: The name of the output file.
        """
        if not urls:
            print("No URLs provided for history processing.")
            return

        print(f"Processing {len(urls)} URLs from history...")
        processed_articles = []
        for url in urls:
            data = self._scrap_and_process_url(url)
            if data:
                processed_articles.append(data)

        if not processed_articles:
            print("Could not process any of the provided URLs.")
            return

        history_df = pd.DataFrame(processed_articles)
        
        history_output_path = os.path.join(os.path.dirname(self.output_path), output_filename)
        
        history_df.to_csv(history_output_path, index=False)
        print(f"History processing complete. Results saved to '{history_output_path}'.")



