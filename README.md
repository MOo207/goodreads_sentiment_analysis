# Goodreads Review Analysis Project

This project performs sentiment analysis on Goodreads book reviews using various machine learning and deep learning techniques. The goal is to classify reviews into positive, neutral, or negative sentiments based on their star ratings and textual content.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Visualizations](#visualizations)
9. [Technologies Used](#technologies-used)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

## Project Overview

This project scrapes book reviews from Goodreads, preprocesses the data, and applies various machine learning and deep learning models to classify the sentiment of reviews. The project includes:

- Web scraping of Goodreads reviews
- Data preprocessing and cleaning
- Feature extraction using multiple techniques
- Model training with traditional ML and deep learning approaches
- Performance evaluation and comparison
- Visualization of results

## Dataset

The dataset consists of 2,292 English book reviews scraped from Goodreads, with ratings converted to sentiment categories:

- **Positive** (4-5 stars): 1,552 reviews
- **Negative** (1-2 stars): 525 reviews
- **Neutral** (3 stars): 215 reviews

The average review length is 2,565.44 characters.

## Project Structure

```
DL/
├── code/
│   ├── app.py              # Main application with data processing and modeling
│   ├── config.py           # Configuration settings
│   ├── requirements.txt    # Python dependencies
│   └── scraper.py          # Goodreads web scraper
├── data/
│   └── best 89 with ratings dataset/
├── presentation/
├── results/
├── venv/
└── visualizations/
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DL
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```

4. Download required NLTK data (done automatically in app.py):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

1. Run the scraper to collect reviews (optional, dataset already included):
   ```bash
   python code/scraper.py
   ```

2. Run the main application for data processing and analysis:
   ```bash
   python code/app.py
   ```

3. View results in the `results/` directory and visualizations in the `visualizations/` directory.

## Methodology

### Data Preprocessing

1. **Language Filtering**: Non-English reviews removed using langdetect
2. **Text Cleaning**: Removal of punctuation, numbers, and special characters
3. **Tokenization**: Breaking text into individual words
4. **Stopword Removal**: Eliminating common words that don't contribute to sentiment
5. **Stemming**: Reducing words to their root forms

### Sentiment Classification

Ratings are converted to sentiment categories:
- 1-2 stars: Negative
- 3 stars: Neutral
- 4-5 stars: Positive

### Feature Extraction

Three approaches are used:
1. **Bag of Words (BOW)**: Traditional word count representation
2. **TF-IDF**: Term frequency-inverse document frequency weighting
3. **Sequential Features**: Integer sequences for deep learning models

### Models

Eight different models are trained and compared:
1. SVM with BOW
2. SVM with TF-IDF
3. Logistic Regression with BOW
4. Logistic Regression with TF-IDF
5. Multinomial Naive Bayes with BOW
6. Multinomial Naive Bayes with TF-IDF
7. LSTM RNN
8. SimpleRNN

## Results

The best performing model was **SVM with TF-IDF** achieving **77.03% accuracy**.

| Model | Accuracy |
|-------|----------|
| SVM (TF-IDF) | 77.03% |
| Logistic Regression (BOW) | 76.74% |
| Multinomial Naive Bayes (BOW) | 76.45% |
| Logistic Regression (TF-IDF) | 74.42% |
| SVM (BOW) | 73.26% |
| LSTM RNN | 70.35% |
| Multinomial Naive Bayes (TF-IDF) | 67.73% |
| SimpleRNN | 67.73% |

## Visualizations

Several visualizations are generated and saved in the `visualizations/` directory:

1. `model_performance_comparison.png` - Bar chart comparing all model performances
2. `detailed_model_comparison.png` - Detailed comparison with model categorization
3. `data_pipeline.png` - Visualization of the data processing pipeline
4. `presentation_visualizations.png` - Additional analysis visualizations

## Technologies Used

- **Python**: Primary programming language
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning models
- **NLTK**: Natural language processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **BeautifulSoup**: Web scraping
- **Selenium**: Browser automation
- **Requests**: HTTP library

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is for educational purposes only.

## Contact

Mohammed Ismail - mohammedismaeal522@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
