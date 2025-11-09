import os

# Configuration file for the Goodreads scraper and sentiment analysis project

# File paths
DEFAULT_DATASET_PATH = r"d:\DL\data\best 89 with ratings dataset"
UNIFIED_DATASET_FILENAME = "unified_goodreads_reviews.csv"

# Web scraping settings
SCRAPER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://www.goodreads.com/',
    'DNT': '1',
    'Connection': 'keep-alive',
}

# Scraper settings
DEFAULT_MAX_PAGES = 5
DEFAULT_MAX_BOOKS = 100
DEFAULT_DELAY_RANGE = (2, 5)
DEFAULT_TIMEOUT = 10
DEFAULT_RETRY_DELAY = 10

# Popular books list URL
POPULAR_BOOKS_URL = 'https://www.goodreads.com/list/show/154948.Most_Popular_Books_on_Goodreads'

# Fallback books (when scraping fails)
FALLBACK_BOOKS = [
    {'title': '1984', 'author': 'George Orwell', 'id': '61439040'},
    {'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald', 'id': '4671'},
    {'title': 'The Catcher in the Rye', 'author': 'J.D. Salinger', 'id': '7178'}
]

# Model parameters
MAX_FEATURES = 10000
MAX_WORDS = 10000
MAX_LENGTH = 1000
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT_RATE = 0.2
DENSE_DROPOUT_RATE = 0.5

# RNN settings
RNN_TYPES = ['LSTM', 'SimpleRNN']
DEFAULT_RNN_TYPE = 'LSTM'

# Training settings
DEFAULT_TEST_SIZE = 0.15
DEFAULT_RANDOM_STATE = 42
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_EARLY_STOPPING_PATIENCE = 3

# Sentiment classification
SENTIMENT_MAPPING = {'negative': 0, 'neutral': 1, 'positive': 2}
RATING_SENTIMENT_THRESHOLDS = {
    'negative': (1, 2),
    'neutral': (3, 3),
    'positive': (4, 5)
}

# Visualization settings
VISUALIZATION_DPI = 300
FIGURE_SIZES = {
    'presentation': (15, 12),
    'model_comparison': (20, 8),
    'detailed_comparison': (16, 10)
}