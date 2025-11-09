# Goodreads Review Analysis Presentation Materials

This document summarizes all the materials created for the Goodreads Review Analysis presentation.

## 1. Dataset Overview

### Statistics
- Total Reviews: 2,292 (after cleaning)
- Unique Sentiments: 3 (Positive, Neutral, Negative)
- Average Review Length: 2,565.44 characters
- Sentiment Distribution:
  - Positive (4-5 stars): 1,552 reviews
  - Negative (1-2 stars): 525 reviews
  - Neutral (3 stars): 215 reviews

### Sample Data
**Before Cleaning:**
```
review                                                rating
"It's written 1948? Clearly History has its twi..."   5.0
"حدثني عن القهر عن الاستعباد عن الذل ثم حدثني ب..."   5.0
"YOU. ARE. THE. DEAD. Oh my God. I got the chil..."   5.0
```

**After Text Preprocessing:**
- Original: "This book was absolutely fantastic! I couldn't put it down."
- Processed: "book absolut fantast could put"

## 2. Methodology

### Data Preprocessing
1. Language filtering: Only English reviews are retained
2. Text cleaning: Removal of punctuation, numbers, and special characters
3. Tokenization: Breaking text into individual words
4. Stopword removal: Common words that don't contribute to sentiment
5. Stemming/Lemmatization: Reducing words to their root forms
6. Negation handling: Preserving sentiment context in negated phrases

### Sentiment Classification
- Converting 5-star ratings to 3 sentiment categories:
  - 1-2 stars: Negative sentiment
  - 3 stars: Neutral sentiment
  - 4-5 stars: Positive sentiment

### Feature Extraction
- Bag of Words (BOW): Traditional word count approach
- TF-IDF: Term frequency-inverse document frequency for word importance
- Sequential features: For deep learning models

### Model Training
- Traditional ML models: Naive Bayes, Logistic Regression, SVM
- Deep Learning models: LSTM and SimpleRNN for sequence modeling
- Cross-validation: Ensuring robust model evaluation

## 3. Flowchart

```
1. Data Collection
   |
   ├── Load 88 CSV files from dataset
   └── Combine into unified dataframe

2. Data Preprocessing
   |
   ├── Remove rows with missing reviews/ratings
   ├── Convert ratings to numeric values
   ├── Clean text data (remove punctuation, numbers)
   ├── Tokenize text
   ├── Remove stopwords
   └── Apply stemming

3. Data Preparation
   |
   ├── Split data into train/test sets
   ├── Encode labels (ratings 1-5)
   ├── Prepare features using:
   |   ├── Bag of Words
   |   ├── TF-IDF
   |   └── Sequences for RNN
   └── Normalize data where needed

4. Model Training & Evaluation
   |
   ├── Train multiple classifiers:
   |   ├── Naive Bayes
   |   ├── Logistic Regression
   |   ├── SVM
   |   └── RNN (LSTM/SimpleRNN)
   ├── Evaluate with accuracy, precision, recall
   └── Compare model performance

5. Results Analysis
   |
   ├── Select best performing model
   ├── Generate classification reports
   └── Create visualizations
```

## 4. Model Comparison Tables

### Model Performance Results
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

### Model Parameters Comparison
| Model Type | Feature Extraction | Key Parameters |
|------------|-------------------|----------------|
| Naive Bayes | Bag of Words | Default parameters |
| Logistic Regression | Bag of Words | max_iter=1000 |
| SVM | Bag of Words | kernel=linear |
| Naive Bayes | TF-IDF | Default parameters |
| Logistic Regression | TF-IDF | max_iter=1000 |
| SVM | TF-IDF | kernel=linear |
| LSTM RNN | Sequences | LSTM_UNITS=128, dropout=0.5 |
| SimpleRNN | Sequences | LSTM_UNITS=128, dropout=0.5 |

## 5. Detailed Analysis

### Performance Statistics
- Mean Accuracy: 72.96%
- Best Accuracy: 77.03% (SVM with TF-IDF)
- Worst Accuracy: 67.73% (Multinomial Naive Bayes with TF-IDF and SimpleRNN)
- Models above 70%: 6 out of 8

### Performance by Model Category
- Bag of Words Models - Mean Accuracy: 75.48%
- TF-IDF Models - Mean Accuracy: 73.06%
- RNN Models - Mean Accuracy: 69.04%

## 6. Dataset Labeling and Augmentation

### Dataset Labeling
The dataset was labeled using the star ratings from Goodreads reviews:
- 1-2 stars: Negative sentiment
- 3 stars: Neutral sentiment
- 4-5 stars: Positive sentiment

### Dataset Augmentation Techniques
1. Language Filtering:
   - Non-English reviews were filtered out using langdetect library
   - This improved data quality by ensuring linguistic consistency

2. Text Preprocessing:
   - Removal of punctuation and numbers
   - Tokenization
   - Stopword removal
   - Stemming

3. Feature Engineering:
   - Bag of Words representation
   - TF-IDF representation
   - Sequential representation for RNN models

4. Data Balancing:
   - SMOTE + Tomek Links applied to handle class imbalance
   - This improved model performance on minority classes

## 7. Visualizations

### Created Visualizations
1. `model_performance_comparison.png`
   - Bar chart comparing all model performances
   - Horizontal bar chart with color coding

2. `detailed_model_comparison.png`
   - Detailed horizontal bar chart with model categorization
   - Color-coded by model type

3. `data_pipeline.png`
   - Visualization of the data processing pipeline
   - Shows the number of reviews at each stage

4. `presentation_visualizations.png`
   - Original visualizations from app.py
   - Sentiment distribution
   - Review length distribution
   - Model comparison
   - Average review length by sentiment

## 8. Enhancement Recommendations

To further improve model performance, consider the following enhancements:

1. **Advanced Text Preprocessing:**
   - Implement spell correction using pyspellchecker
   - Convert emojis to text representations
   - Handle negations more effectively

2. **Improved Feature Extraction:**
   - Use Complement Naive Bayes for imbalanced datasets
   - Expand TF-IDF features to 15K with n-grams up to 3
   - Implement Word2Vec embeddings to capture semantic relationships

3. **Ensemble Methods:**
   - Create soft voting ensembles combining NB, LR, and SVM models
   - Implement stacked ensembles with meta-learners

4. **Hyperparameter Optimization:**
   - Perform systematic hyperparameter tuning using GridSearchCV
   - Use cross-validation for robust evaluation

5. **Advanced Models:**
   - Implement Bidirectional LSTM for better sequence modeling
   - Try CNN models for text classification
   - Experiment with hybrid CNN-LSTM architectures

## 9. Conclusion

- Successfully classified Goodreads reviews into 3 sentiment categories
- SVM with TF-IDF achieved the best performance (77.03% accuracy)
- Traditional ML models outperformed deep learning models
- Language filtering significantly improved data quality
- Future work: Implement enhancement recommendations

## 10. Files Generated

All presentation materials have been organized in the following files:

1. `flowchart.txt` - Data processing pipeline flowchart
2. `methodology.txt` - Detailed methodology description
3. `dataset_statistics.csv` - Dataset metrics
4. `model_results.csv` - Model performance results
5. `presentation_visualizations.png` - Data visualizations
6. `model_performance_comparison.png` - Model comparison charts
7. `detailed_model_comparison.png` - Detailed model comparison chart
8. `data_pipeline.png` - Data processing pipeline visualization
9. `presentation_summary.txt` - Comprehensive presentation summary
10. `presentation_slides.txt` - Slide-by-slide presentation outline