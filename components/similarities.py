import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import ast

class SimilarityCalculator:
    """
    Calculates TF-IDF based similarities between a corpus and a user's history,
    providing recommendations.
    """

    def __init__(self, processed_corpus_path, processed_history_path, model_dir='models'):
        """
        Initializes the SimilarityCalculator.

        :param processed_corpus_path: Path to the processed corpus CSV file.
        :param processed_history_path: Path to the processed history CSV file.
        :param model_dir: Directory to save/load the TF-IDF model and vectors.
        """
        self.corpus_path = processed_corpus_path
        self.history_path = processed_history_path
        self.model_dir = model_dir
        
        # Ensure the model directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Define paths for saved model components
        self.vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        self.corpus_vectors_path = os.path.join(self.model_dir, 'corpus_tfidf_vectors.joblib')
        self.corpus_data_path = os.path.join(self.model_dir, 'corpus_data.joblib')

        self.vectorizer = None
        self.corpus_vectors = None
        self.corpus_data = None

    def _load_and_preprocess_data(self, file_path):
        """Loads data and preprocesses the text column."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        def safe_literal_eval(x):
            if pd.isna(x) or not isinstance(x, str):
                return ''
            try:
                return ' '.join(ast.literal_eval(x))
            except (ValueError, SyntaxError):
                return ''

        df['processed_text'] = df['lemmatized'].apply(safe_literal_eval)
        return df

    def fit_corpus(self, force_refit=False):
        """
        Fits the TfidfVectorizer on the corpus and saves the model and vectors.
        
        :param force_refit: If True, re-fits the model even if a saved one exists.
        """
        if not force_refit and os.path.exists(self.vectorizer_path):
            print("TF-IDF model already exists. Loading from disk.")
            self.load_model()
            return

        print("Fitting TF-IDF model on the corpus...")
        self.corpus_data = self._load_and_preprocess_data(self.corpus_path)
        
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.corpus_vectors = self.vectorizer.fit_transform(self.corpus_data['processed_text'])
        
        # Save the fitted components
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.corpus_vectors, self.corpus_vectors_path)
        joblib.dump(self.corpus_data[['url', 'title']], self.corpus_data_path)
        
        print(f"TF-IDF model and vectors saved to '{self.model_dir}'.")

    def load_model(self):
        """Loads the pre-fitted TF-IDF vectorizer, vectors, and data from disk."""
        if not all(os.path.exists(p) for p in [self.vectorizer_path, self.corpus_vectors_path, self.corpus_data_path]):
            raise FileNotFoundError("Model files not found. Please run `fit_corpus()` first.")
            
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.corpus_vectors = joblib.load(self.corpus_vectors_path)
        self.corpus_data = joblib.load(self.corpus_data_path)
        print("Successfully loaded TF-IDF model from disk.")

    def get_recommendations(self, top_k=10):
        """
        Calculates similarity between history and corpus, and returns top K recommendations.

        :param top_k: The number of top articles to recommend.
        :return: A list of tuples, where each tuple is (url, title, score).
        """
        if self.vectorizer is None:
            self.load_model()

        print("Processing history to generate recommendations...")
        history_data = self._load_and_preprocess_data(self.history_path)
        
        if history_data.empty:
            print("History is empty. No recommendations to generate.")
            return []

        history_urls = set(history_data['url'])
        history_vectors = self.vectorizer.transform(history_data['processed_text'])

        similarity_matrix = cosine_similarity(history_vectors, self.corpus_vectors)

        # Average the similarity scores across all history articles
        corpus_scores = similarity_matrix.mean(axis=0)

        recommendation_df = self.corpus_data.copy()
        recommendation_df['score'] = corpus_scores
        recommendation_df = recommendation_df.sort_values('score', ascending=False).drop_duplicates('url', keep='first')
        recommendation_df = recommendation_df[~recommendation_df['url'].isin(history_urls)]
        
        top_articles = recommendation_df.head(top_k)

        recommendations = [
            (row['url'], row['title'], row['score'])
            for index, row in top_articles.iterrows()
        ]
        
        return recommendations


