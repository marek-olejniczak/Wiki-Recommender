import os
import pandas as pd
import glob

class ArticleMerger:
    """
    A class to merge scraped Wikipedia articles from multiple CSV files into a single corpus file.
    """

    def __init__(self, data_directory, corpus_file_name='wiki_articles_corpus.csv'):
        """
        Initializes the ArticleMerger.

        :param data_directory: The directory where the individual article CSV files are stored.
        :param corpus_file_name: The name of the merged corpus file.
        """
        self.data_directory = data_directory
        self.corpus_file_path = os.path.join(self.data_directory, corpus_file_name)
        self.source_files_pattern = os.path.join(self.data_directory, 'wiki_articles_*.csv')
        self.columns = ['url', 'title', 'text', 'domain', 'timestamp', 'word_count', 'char_count', 'content_hash']

    def merge_articles(self):
        """
        Merges all unique articles from source CSVs into the corpus file.
        :return: True if successful, False otherwise.
        """
        try:
            existing_hashes = set()

            # Read existing content hashes
            if os.path.exists(self.corpus_file_path):
                try:
                    corpus_df = pd.read_csv(self.corpus_file_path)
                    if 'content_hash' in corpus_df.columns:
                        existing_hashes.update(corpus_df['content_hash'].astype(str))
                    print(f"Found {len(existing_hashes)} existing articles in '{self.corpus_file_path}'.")
                except pd.errors.EmptyDataError:
                    print(f"Corpus file '{self.corpus_file_path}' is empty.")
            
            source_files = glob.glob(self.source_files_pattern)
            
            # Exclude the corpus file
            if self.corpus_file_path in source_files:
                source_files.remove(self.corpus_file_path)

            if not source_files:
                print("No source article files found to merge.")
                return True

            print(f"Found {len(source_files)} source files to process.")

            new_articles = []

            for file_path in source_files:
                try:
                    df = pd.read_csv(file_path)
                    if 'content_hash' not in df.columns:
                        print(f"Skipping file without 'content_hash': {file_path}")
                        continue
                    
                    for _, row in df.iterrows():
                        content_hash = str(row['content_hash'])
                        if content_hash not in existing_hashes:
                            new_articles.append(row)
                            existing_hashes.add(content_hash)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

            if not new_articles:
                print("No new articles to add.")
                return True

            print(f"Found {len(new_articles)} new unique articles to add.")

            new_articles_df = pd.DataFrame(new_articles, columns=self.columns)

            if not os.path.exists(self.corpus_file_path) or os.path.getsize(self.corpus_file_path) == 0:
                new_articles_df.to_csv(self.corpus_file_path, index=False, header=True)
                print(f"Created corpus file and added {len(new_articles_df)} articles.")
            else:
                new_articles_df.to_csv(self.corpus_file_path, mode='a', index=False, header=False)
                print(f"Appended {len(new_articles_df)} new articles to the corpus file.")
            
            return True
        except Exception as e:
            print(f"An error occurred during the merge process: {e}")
            return False

    def cleanup_source_files(self):
        """
        Deletes all source article files except for the corpus file.
        """
        print("Cleaning up source files...")
        source_files = glob.glob(self.source_files_pattern)

        if self.corpus_file_path in source_files:
            source_files.remove(self.corpus_file_path)

        if not source_files:
            print("No source files to clean up.")
            return

        for file_path in source_files:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")


