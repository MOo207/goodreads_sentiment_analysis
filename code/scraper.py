import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from datetime import datetime
import logging
from config import SCRAPER_HEADERS, DEFAULT_MAX_PAGES, DEFAULT_MAX_BOOKS, DEFAULT_DELAY_RANGE, DEFAULT_TIMEOUT, DEFAULT_RETRY_DELAY, POPULAR_BOOKS_URL, FALLBACK_BOOKS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

def get_headers():
    """Return a dictionary of headers to mimic a browser request"""
    return SCRAPER_HEADERS

def scrape_goodreads_reviews(book_id, max_pages=DEFAULT_MAX_PAGES):
    """
    Scrape book reviews from Goodreads
    
    Args:
        book_id (str): The Goodreads book ID (e.g., '5907' for The Hobbit)
        max_pages (int): Maximum number of review pages to scrape
        
    Returns:
        list: List of dictionaries containing review texts and ratings
    """
    base_url = f'https://www.goodreads.com/book/show/{book_id}'
    reviews = []
    session = requests.Session()
    
    for page in range(1, max_pages + 1):
        try:
            url = f'{base_url}?page={page}'
            logging.info(f'Scraping page {page}...')
            
            # Add random delay between requests
            time.sleep(random.uniform(*DEFAULT_DELAY_RANGE))
            
            response = session.get(url, headers=get_headers(), timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            # Check for CAPTCHA or access denied
            if "captcha" in response.text.lower() or "access denied" in response.text.lower():
                logging.warning("CAPTCHA or access denied detected. Try again later or use a proxy.")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- FIX 1: The main review container is an <article> tag with class "ReviewCard" ---
            review_containers = soup.select('article.ReviewCard')
                
            if not review_containers:
                logging.warning("No reviews found on page %s. The page structure might have changed.", page)
                logging.debug("Page content: %s", response.text[:500])
                break
                
            for container in review_containers:
                try:
                    # Remove any script or style elements
                    for element in container(['script', 'style', 'noscript', 'button', 'a']):
                        element.decompose()
                    
                    # --- FIX 2: Target the review text container specifically ---
                    # This prevents grabbing user names, dates, and tags as part of the review.
                    review_text_element = container.select_one('section.ReviewText__content')
                    if review_text_element:
                         review = ' '.join(review_text_element.stripped_strings)
                    else:
                         # Fallback to original method if new selector fails
                         review = ' '.join(container.stripped_strings)
                    
                    # Initialize rating as None
                    rating = None
                    
                    # --- FIX 3: Replace brittle rating logic with a single, robust selector ---
                    # Based on example.html, the rating is in a span like:
                    # <span aria-label="Rating 5 out of 5" ...>
                    
                    # This selector finds a span with class "RatingStars" that has an aria-label starting with "Rating"
                    rating_span = container.select_one('span.RatingStars[aria-label^="Rating"]')
                    
                    if rating_span:
                        aria_label = rating_span.get('aria-label', '')
                        # Extract the first number (the rating)
                        rating_match = re.search(r'(\d+)', aria_label)
                        if rating_match:
                            try:
                                rating = float(rating_match.group(1))
                                if not (1 <= rating <= 5):
                                    rating = None  # Invalid rating number
                            except (ValueError, IndexError):
                                pass # Failed to parse
                    
                    # --- End of new rating logic ---
                    
                    if review and len(review) > 50:  # Only add meaningful reviews
                        # Ensure every review has a rating field, even if None
                        reviews.append({
                            'review': review,
                            'rating': rating  # This will be None if no rating found
                        })
                        logging.debug("Added review: %s...", review[:100])
                        if rating:
                            logging.debug("Rating: %s", rating)
                        else:
                            logging.debug("No rating found for review")
                        
                except Exception as e:
                    logging.error("Error processing review container: %s", str(e))
                    # Still add the review even if there's an error processing rating
                    try:
                        review_text_element = container.select_one('section.ReviewText__content')
                        if review_text_element:
                            review = ' '.join(review_text_element.stripped_strings)
                        else:
                            review = ' '.join(container.stripped_strings)
                            
                        if review and len(review) > 50:
                            reviews.append({
                                'review': review,
                                'rating': None  # Add review with None rating in case of error
                            })
                    except:
                        pass
                    continue
            
            logging.info("Found %d reviews on page %d", len(review_containers), page)
            
            # Check if there's a next page
            next_page = soup.select_one('a.next_page')
            if not next_page or ('class' in next_page.attrs and 'disabled' in next_page.attrs['class']):
                break
                
        except requests.exceptions.RequestException as e:
            logging.error("Error scraping page %d: %s", page, str(e))
            time.sleep(DEFAULT_RETRY_DELAY)
            continue
        except Exception as e:
            logging.error("Unexpected error on page %d: %s", page, str(e), exc_info=True)
            continue
    
    return reviews

def scrape_popular_books(list_url, max_books=DEFAULT_MAX_BOOKS):
    """
    Scrape popular books from a Goodreads list
    
    Args:
        list_url (str): URL of the Goodreads list
        max_books (int): Maximum number of books to scrape
        
    Returns:
        list: List of dictionaries containing book information (title, author, id)
    """
    try:
        logging.info(f"Scraping popular books from: {list_url}")
        
        response = requests.get(list_url, headers=get_headers(), timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        books = []
        
        # Find all book rows in the list
        book_rows = soup.select('tr[itemtype="http://schema.org/Book"]')
        
        for row in book_rows[:max_books]:
            try:
                # Extract book title
                title_elem = row.select_one('a.bookTitle span')
                title = title_elem.text.strip() if title_elem else 'Unknown Title'
                
                # Extract author
                author_elem = row.select_one('a.authorName span')
                author = author_elem.text.strip() if author_elem else 'Unknown Author'
                
                # Extract book ID from the book URL
                book_link = row.find('a', class_='bookTitle')
                if book_link and 'href' in book_link.attrs:
                    book_url = str(book_link['href'])  # Convert to string to handle AttributeValueList
                    # Extract ID from URL (format: /book/show/12345-title)
                    match = re.search(r'/book/show/(\d+)', book_url)
                    if match:
                        book_id = match.group(1)
                        books.append({
                            'title': title,
                            # 'author': author,
                            'id': book_id,
                            'url': f"https://www.goodreads.com{book_url}"
                        })
            except Exception as e:
                logging.error(f"Error processing book row: {str(e)}")
                continue
        
        logging.info(f"Successfully scraped {len(books)} popular books")
        return books
        
    except Exception as e:
        logging.error(f"Error scraping popular books: {str(e)}")
        return []

def classify_sentiment(rating):
    """
    Classify sentiment based on rating value
    
    Args:
        rating (float): The rating value (1-5 scale)
        
    Returns:
        str: Sentiment classification ("negative", "neutral", or "positive")
    """
    if rating is None:
        return None
    elif 1 <= rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    elif 4 <= rating <= 5:
        return "positive"
    else:
        return None

def main():
    # Scrape popular books from the Goodreads list
    books = scrape_popular_books(POPULAR_BOOKS_URL, max_books=DEFAULT_MAX_BOOKS)
    
    if not books:
        logging.warning("No books were scraped from the popular list. Using default books as fallback.")
        books = FALLBACK_BOOKS
    
    book_ids = [book['id'] for book in books]  # Extract just the IDs for the scraping function
    
    try:
        logging.info("Starting Goodreads scraper...")
        
        for book in books:
            book_id = book['id']
            book_title = book['title']
            # book_author = book['author']
            
            # logging.info(f"Scraping reviews for: {book_title} by {book_author} (ID: {book_id})")
            reviews_data = scrape_goodreads_reviews(book_id, max_pages=DEFAULT_MAX_PAGES)
            
            if reviews_data:
                # Extract reviews, ratings, and classify sentiments
                reviews = [item['review'] for item in reviews_data]
                ratings = [item['rating'] for item in reviews_data]
                sentiments = [classify_sentiment(item['rating']) for item in reviews_data]
                
                # Create DataFrame with additional book information, ratings, and sentiments
                df = pd.DataFrame({'review': reviews, 'rating': ratings, 'sentiment': sentiments})
                df['book_title'] = book_title
                # df['book_author'] = book_author
                df['book_id'] = book_id
                
                # Create a clean filename with book title and author
                safe_title = "".join([c if c.isalnum() else "_" for c in book_title])
                # safe_author = "".join([c if c.isalnum() else "_" for c in book_author])
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'goodreads_reviews_{safe_title}_{timestamp}.csv'
                
                # Save to CSV
                df.to_csv(filename, index=False, encoding='utf-8')
                logging.info("Successfully saved %d reviews to %s", len(reviews), filename)
            else:
                logging.warning("No reviews were scraped for book ID: %s", book_id)
            
            # Add a small delay between processing different books
            time.sleep(DEFAULT_DELAY_RANGE[1])
            
    except Exception as e:
        logging.error("Fatal error in main: %s", str(e), exc_info=True)
    finally:
        logging.info("Scraping completed for all books.")

if __name__ == "__main__":
    main()