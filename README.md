# Wiki-Recommender

A tool that recommends Wikipedia articles based on your browsing history. It works by comparing a corpus of scraped Wikipedia articles with a list of URLs you've visited.

## How It Works

The project follows these main steps to generate recommendations:

1.  **Scrape Wikipedia Articles**: The `components/crawler.py` contains an asynchronous web crawler that scrapes articles from Wikipedia. It starts from a given list of URLs and follows links to a specified depth, collecting a target number of articles. The scraped articles are saved as individual CSV files in the `data/` directory.

2.  **Merge Articles**: The `components/articles.py` script merges the individual CSV files into a single large corpus, `data/wiki_articles_corpus.csv`. This corpus is the knowledge base for the recommender.

3.  **Process Corpus**: The text in the corpus is processed using `components/processor.py`. This involves tokenization, stemming, and lemmatization to prepare the text for machine learning. The processed corpus is saved to `data/processed_articles.csv`.

4.  **Process Browsing History**: A list of Wikipedia URLs, representing your browsing history, is scraped and processed using the same text processing pipeline.

5.  **Build TF-IDF Model**: A TF-IDF (Term Frequency-Inverse Document Frequency) model is built from the processed corpus using `components/similarities.py`. This model represents the importance of words in the articles. If a model already exists, it's loaded from the `models/` directory.

6.  **Get Recommendations**: The TF-IDF model is used to calculate the cosine similarity between your browsing history and each article in the corpus. The articles with the highest similarity scores are returned as recommendations.

## How to Use

The entire process is demonstrated in the `recommender.ipynb` Jupyter Notebook. To get recommendations, follow these steps:

1.  **Setup**: Make sure you have the required libraries installed. You can find them in the imports section of the notebook.
2.  **Open and Run `recommender.ipynb`**: Open the notebook in a Jupyter environment.
3.  **Define Your History**: In the notebook, you can change the `history_urls_to_process` list to include the Wikipedia articles you are interested in.
4.  **Run the Cells**: Execute the cells in the notebook in order. The notebook will:
    -   Optionally scrape more articles to expand the corpus.
    -   Merge and process the corpus.
    -   Process your browsing history.
    -   Build or load the TF-IDF model.
    -   Output the top recommended articles based on your history.
