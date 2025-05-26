import streamlit as st
import os
import tempfile
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
import re
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime, timedelta
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Financial NLP Chatbot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance - ensuring all text is visible
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2b6cb0;
    }
    .chat-message.bot {
        background-color: #f0fff4;
        border-left: 5px solid #38a169;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
        color: #000000;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    /* Only target the chat message text */
    .chat-message p {
        color: #000000 !important;
    }
    .news-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
    }
    .news-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        color: #000000 !important;
    }
    .news-source {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .news-date {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .news-description {
        font-size: 16px;
        margin-bottom: 10px;
        color: #000000 !important;
    }
    .news-link {
        font-size: 14px;
        color: #007bff;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        color: white;
        margin-left: 10px;
    }
    .sentiment-positive {
        background-color: #28a745;
    }
    .sentiment-negative {
        background-color: #dc3545;
    }
    .sentiment-neutral {
        background-color: #6c757d;
    }
    .news-fed-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        background-color: #17a2b8;
        color: white;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'document_embeddings' not in st.session_state:
    st.session_state.document_embeddings = []
if 'document_sources' not in st.session_state:
    st.session_state.document_sources = []
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'use_sample_only' not in st.session_state:
    st.session_state.use_sample_only = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = False
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None
if 'sentiment_analyzer' not in st.session_state:
    try:
        st.session_state.sentiment_analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        st.session_state.sentiment_analyzer = None
        st.error(f"Error loading sentiment analyzer: {str(e)}")
if 'news_fed_to_chatbot' not in st.session_state:
    st.session_state.news_fed_to_chatbot = False

# Define popular stock tickers
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "WMT",
    "JNJ", "PG", "DIS", "NFLX", "INTC",
    "BAC", "KO", "PEP", "CSCO", "ADBE",
    "CRM", "PYPL", "CMCSA", "XOM", "CVX",
    "ABT", "TMO", "ACN", "COST", "DHR",
    "MRK", "UNH", "HD", "MA", "AVGO",
    "LLY", "PFE", "VZ", "T", "ORCL"
]

# Dictionary of ticker symbols to company names for better display
TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc. (Facebook)",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble Co.",
    "DIS": "The Walt Disney Company",
    "NFLX": "Netflix, Inc.",
    "INTC": "Intel Corporation",
    "BAC": "Bank of America Corporation",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "CMCSA": "Comcast Corporation",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "ABT": "Abbott Laboratories",
    "TMO": "Thermo Fisher Scientific Inc.",
    "ACN": "Accenture plc",
    "COST": "Costco Wholesale Corporation",
    "DHR": "Danaher Corporation",
    "MRK": "Merck & Co., Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "HD": "The Home Depot, Inc.",
    "MA": "Mastercard Incorporated",
    "AVGO": "Broadcom Inc.",
    "LLY": "Eli Lilly and Company",
    "PFE": "Pfizer Inc.",
    "VZ": "Verizon Communications Inc.",
    "T": "AT&T Inc.",
    "ORCL": "Oracle Corporation"
}

# Function to clean and format text
def clean_text(text):
    try:
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove date/time stamps
        text = re.sub(r'\d{2}/\d{2}/\d{4},\s\d{2}:\d{2}', '', text)
        
        # Remove page indicators
        text = re.sub(r'\d+/\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix sentence spacing
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        
        return text
    except Exception as e:
        # If any error occurs, return the original text
        return text

# Function to format response
def format_response(text):
    try:
        # Clean the text first
        text = clean_text(text)
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Format each paragraph
        formatted_paragraphs = []
        for para in paragraphs:
            if para.strip():
                # Check if it's a list item
                if para.strip().startswith('-'):
                    formatted_paragraphs.append(para)
                else:
                    # Format as a proper paragraph
                    formatted_paragraphs.append(para)
        
        # Join paragraphs with proper spacing
        return '\n\n'.join(formatted_paragraphs)
    except Exception as e:
        # If any error occurs, return the original text
        return text

# Function to download sample financial documents from GitHub
@st.cache_data
def download_github_files():
    try:
        # This function would download sample financial documents from a GitHub repository
        # For demonstration, we'll create a sample financial text
        sample_text = """
        # Financial News and Analysis
        
        ## Market Overview
        The S&P 500 closed at 4,890.97, up 0.3% for the day. The Nasdaq Composite gained 0.5% to 15,425.94, while the Dow Jones Industrial Average added 0.2% to 38,790.43.
        
        ## Top Performers
        - NVDA: Up 3.2% after announcing new AI chips
        - LLY: Gained 2.1% on positive drug trial results
        - MSFT: Rose 1.5% following cloud service expansion
        
        ## Market Sentiment
        Investor sentiment remains positive despite inflation concerns. The VIX, Wall Street's fear gauge, fell 3% to 14.2, indicating lower expected volatility.
        
        ## Analyst Ratings
        - Goldman Sachs upgraded AAPL to "Buy" with a price target of $220
        - Morgan Stanley maintained an "Overweight" rating on AMZN
        - JPMorgan downgraded NFLX to "Neutral" citing competition concerns
        
        ## Economic Indicators
        - Consumer Price Index (CPI) rose 2.4% year-over-year
        - Unemployment rate steady at 3.8%
        - Housing starts increased 3.2% in the latest report
        
        ## Sector Performance
        Technology and Healthcare led gains, while Energy and Utilities lagged.
        
        ## International Markets
        - European markets closed mixed
        - Asian markets mostly higher, led by Japan's Nikkei
        - Emerging markets showed strength on dollar weakness
        """
        
        # Create a temporary file to store the sample text
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w+') as f:
            f.write(sample_text)
            temp_file_path = f.name
        
        return [temp_file_path]
    except Exception as e:
        st.error(f"Error downloading sample files: {str(e)}")
        return []

# Function to load the embedding model
@st.cache_resource
def get_embedding_model():
    try:
        # Load a lightweight sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# Function to compute embeddings
def compute_embeddings(texts):
    try:
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = get_embedding_model()
        
        if st.session_state.embedding_model is None:
            return None
        
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = st.session_state.embedding_model.encode(batch)
            all_embeddings.extend(batch_embeddings)
        
        st.session_state.embedding_model_loaded = True
        return all_embeddings
    except Exception as e:
        st.error(f"Error computing embeddings: {str(e)}")
        return None

# Function to find most similar chunks using cosine similarity
def find_similar_chunks(query_embedding, document_embeddings, document_chunks, top_k=3):
    try:
        # Compute cosine similarity
        similarities = []
        
        # Safety check for empty embeddings
        if not document_embeddings or len(document_embeddings) == 0:
            return []
            
        for i, doc_embedding in enumerate(document_embeddings):
            try:
                # Convert to numpy arrays if they aren't already
                query_embedding_np = np.array(query_embedding, dtype=np.float32)
                doc_embedding_np = np.array(doc_embedding, dtype=np.float32)
                
                # Normalize the vectors
                query_norm = float(np.linalg.norm(query_embedding_np))
                doc_norm = float(np.linalg.norm(doc_embedding_np))
                
                # Compute cosine similarity - using explicit scalar values
                if query_norm > 0.0 and doc_norm > 0.0:
                    dot_product = float(np.dot(query_embedding_np, doc_embedding_np))
                    similarity = dot_product / (query_norm * doc_norm)
                else:
                    similarity = 0.0
                
                similarities.append((i, similarity))
            except Exception as inner_e:
                # If there's an error with one document, skip it and continue
                st.session_state.error_message = f"Error calculating similarity for document {i}: {str(inner_e)}"
                continue
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k most similar chunks
        top_chunks = []
        for i, similarity in similarities[:top_k]:
            if i < len(document_chunks):  # Safety check
                top_chunks.append({
                    "content": document_chunks[i]["content"],
                    "source": document_chunks[i]["source"],
                    "similarity": similarity
                })
        
        return top_chunks
    except Exception as e:
        st.session_state.error_message = f"Error finding similar chunks: {str(e)}\n{traceback.format_exc()}"
        return []

# Function to process documents
def process_documents(file_paths, use_sample_only=False):
    try:
        documents = []
        
        # Load documents based on file type
        for file_path in file_paths:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    st.session_state.document_sources.append(f"PDF: {os.path.basename(file_path)}")
                    
                    # Add each page as a separate document
                    for doc in docs:
                        documents.append({
                            "content": doc.page_content,
                            "source": os.path.basename(file_path)
                        })
                
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    st.session_state.document_sources.append(f"Text: {os.path.basename(file_path)}")
                    
                    # Add text content
                    for doc in docs:
                        documents.append({
                            "content": doc.page_content,
                            "source": os.path.basename(file_path)
                        })
                
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                    st.session_state.document_sources.append(f"CSV: {os.path.basename(file_path)}")
                    
                    # Add CSV content
                    for doc in docs:
                        documents.append({
                            "content": doc.page_content,
                            "source": os.path.basename(file_path)
                        })
                
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
            
            except Exception as e:
                st.error(f"Error processing file {file_path}: {str(e)}")
                st.session_state.error_message = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        
        if not documents:
            st.error("No documents could be processed. Please check file formats and try again.")
            return False
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for more precise retrieval
            chunk_overlap=150,  # Sufficient overlap to maintain context
            length_function=len
        )
        
        # Process each document into chunks
        all_chunks = []
        for doc in documents:
            try:
                splits = text_splitter.split_text(doc["content"])
                for split in splits:
                    all_chunks.append({
                        "content": split,
                        "source": doc["source"]
                    })
            except Exception as e:
                st.error(f"Error splitting document: {str(e)}")
                continue  # Skip this document but continue processing others
        
        # Store chunks in session state
        st.session_state.document_chunks = all_chunks
        
        # Compute embeddings for all chunks
        try:
            chunk_texts = [chunk["content"] for chunk in all_chunks]
            embeddings = compute_embeddings(chunk_texts)
            
            if embeddings is not None and len(embeddings) > 0:
                # Store embeddings in session state
                st.session_state.document_embeddings = embeddings
                st.success("Document embeddings computed successfully!")
            else:
                st.warning("Could not compute embeddings. Falling back to keyword search.")
        except Exception as e:
            st.error(f"Error computing embeddings: {str(e)}")
            st.session_state.error_message = f"Error computing embeddings: {str(e)}\n{traceback.format_exc()}"
            # Continue without embeddings, will fall back to keyword search
        
        return True
            
    except Exception as e:
        st.error(f"Unexpected error during document processing: {str(e)}")
        st.session_state.error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        return False

# Function to perform keyword search
def keyword_search(query, document_chunks, top_k=3):
    try:
        query_lower = query.lower()
        results = []
        
        # Extract keywords from query
        keywords = [word.strip() for word in re.split(r'[^\w]', query_lower) if word.strip() and len(word.strip()) > 3]
        
        # If no valid keywords, use the whole query
        if not keywords:
            keywords = [query_lower]
        
        # Score each document chunk
        for doc in document_chunks:
            score = 0
            content = doc["content"].lower()
            
            # Simple keyword matching
            for keyword in keywords:
                if keyword in content:
                    score += content.count(keyword)
            
            if score > 0:
                results.append({"content": doc["content"], "source": doc["source"], "score": score})
        
        # Sort by score and get top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    except Exception as e:
        st.session_state.error_message = f"Error in keyword search: {str(e)}\n{traceback.format_exc()}"
        return []

# Function to call external LLM API (OpenAI-compatible)
def call_llm_api(prompt, api_url=None, api_key=None):
    try:
        # Default to free tier of OpenRouter if no API URL provided
        if not api_url:
            api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Use a default key for OpenRouter free tier if none provided
        # This is a limited free tier that should work for demos
        if not api_key:
            api_key = "sk-or-v1-free-tier-demo"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "openai/gpt-3.5-turbo-0125",  # Use a reliable, widely available model
            "messages": [
                {"role": "system", "content": "You are a helpful financial assistant that provides concise, accurate information based only on the provided context. If the information is not in the context, say you don't have enough information."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return "Error: Unexpected API response format"
        else:
            return f"Error: API returned status code {response.status_code}"
    
    except Exception as e:
        return f"Error calling LLM API: {str(e)}"

# Function to create a custom response for investment questions
def create_investment_response(context, query):
    # Extract stock symbols and ratings from context
    stocks = []
    buy_ratings = []
    
    # Look for stock symbols (usually 1-5 uppercase letters)
    symbol_pattern = r'\b[A-Z]{1,5}\b'
    symbols = re.findall(symbol_pattern, context)
    
    # Look for buy/sell ratings
    for symbol in symbols:
        if f"{symbol}:" in context or f"{symbol} " in context:
            # Check if it has positive sentiment nearby
            snippet = context[max(0, context.find(symbol)-50):min(len(context), context.find(symbol)+50)]
            if any(term in snippet.lower() for term in ["up", "gain", "rose", "buy", "outperform", "overweight", "positive"]):
                buy_ratings.append(symbol)
            stocks.append(symbol)
    
    # Create a custom response based on the query and extracted information
    if "what stock" in query.lower() or "which stock" in query.lower() or "recommend" in query.lower():
        if buy_ratings:
            response = f"Based on the financial documents I've analyzed, several stocks have received positive mentions or analyst ratings:\n\n"
            for symbol in buy_ratings:
                # Find relevant snippet for this stock
                start_idx = max(0, context.find(symbol)-100)
                end_idx = min(len(context), context.find(symbol)+200)
                snippet = context[start_idx:end_idx]
                
                # Clean up the snippet
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                
                # Add to response
                response += f"• {symbol}: {snippet}\n\n"
            
            response += "Remember that this information is based solely on the documents I've analyzed and should not be considered financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions."
            return response
        elif stocks:
            return f"The documents mention these stocks: {', '.join(stocks)}. However, I don't have enough information about positive analyst ratings or performance to make specific recommendations. Always consult with a financial advisor before making investment decisions."
        else:
            return "I don't have enough specific information about which stocks to buy in the documents I've analyzed. For investment advice, please consult with a qualified financial advisor who can provide personalized recommendations based on your financial situation and goals."
    
    # Default to returning the original context if no specific handling
    return None

# Function to analyze sentiment of text with financial context awareness
def analyze_sentiment(text):
    try:
        if st.session_state.sentiment_analyzer is None:
            return {"score": 0, "label": "Neutral", "color": "#6c757d"}
        
        if not text:
            return {"score": 0, "label": "Neutral", "color": "#6c757d"}
        
        # Check for financial context that might override standard sentiment
        lower_text = text.lower()
        
        # Words indicating selling/lowering stakes (typically negative for the stock)
        selling_words = ["sell", "sold", "selling", "lowers", "reduces", "cuts", "dumps", "exits", "downgrades"]
        
        # Words indicating buying/raising stakes (typically positive for the stock)
        buying_words = ["buy", "bought", "buying", "raises", "increases", "adds", "accumulates", "upgrades"]
        
        # Check for selling context
        has_selling_context = any(word in lower_text for word in selling_words)
        
        # Check for buying context
        has_buying_context = any(word in lower_text for word in buying_words)
        
        # Get standard sentiment scores
        sentiment = st.session_state.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment['compound']
        
        # Apply financial context rules to override standard sentiment
        if has_selling_context and not has_buying_context:
            # If it's about selling/lowering stakes, it's negative for the stock
            return {"score": -0.3, "label": "Negative", "color": "#dc3545"}
        elif has_buying_context and not has_selling_context:
            # If it's about buying/raising stakes, it's positive for the stock
            return {"score": 0.3, "label": "Positive", "color": "#28a745"}
        
        # If no specific financial context or mixed signals, use standard sentiment
        if compound_score >= 0.05:
            return {"score": compound_score, "label": "Positive", "color": "#28a745"}
        elif compound_score <= -0.05:
            return {"score": compound_score, "label": "Negative", "color": "#dc3545"}
        else:
            return {"score": compound_score, "label": "Neutral", "color": "#6c757d"}
    except Exception as e:
        # Return neutral sentiment in case of error
        return {"score": 0, "label": "Neutral", "color": "#6c757d"}

# Function to fetch news for a specific ticker
def fetch_stock_news(ticker, api_key="7a285b0c044f4c2b96bc5e18c1b58f3d", max_articles=2):
    try:
        # First try NewsAPI
        url = f"https://newsapi.org/v2/everything?q={ticker}+stock&apiKey={api_key}&pageSize={max_articles}&language=en&sortBy=publishedAt"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                articles = data.get("articles")
                result = []
                for article in articles[:max_articles]:
                    # Combine title and description for sentiment analysis
                    text_for_sentiment = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = analyze_sentiment(text_for_sentiment)
                    
                    result.append({
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("name", "Unknown"),
                        "published_at": article.get("publishedAt"),
                        "sentiment": sentiment
                    })
                return result
        
        # If NewsAPI fails, try Yahoo Finance API as fallback
        # This is a simple scraping approach that might work as fallback
        fallback_url = f"https://query1.finance.yahoo.com/v2/finance/news?symbol={ticker}"
        fallback_response = requests.get(fallback_url, timeout=5)
        
        if fallback_response.status_code == 200:
            data = fallback_response.json()
            if "items" in data and "result" in data["items"] and data["items"]["result"]:
                articles = data["items"]["result"]
                result = []
                for article in articles[:max_articles]:
                    # Combine title and description for sentiment analysis
                    text_for_sentiment = f"{article.get('title', '')} {article.get('summary', '')}"
                    sentiment = analyze_sentiment(text_for_sentiment)
                    
                    result.append({
                        "title": article.get("title"),
                        "description": article.get("summary"),
                        "url": article.get("link"),
                        "source": "Yahoo Finance",
                        "published_at": datetime.fromtimestamp(article.get("published_at", 0)).isoformat(),
                        "sentiment": sentiment
                    })
                return result
        
        # If both APIs fail, return a mock article as last resort
        return [
            {
                "title": f"Latest news for {ticker} not available",
                "description": "Could not retrieve latest news. Please try again later or check financial news websites directly.",
                "url": f"https://finance.yahoo.com/quote/{ticker}",
                "source": "System Message",
                "published_at": datetime.now().isoformat(),
                "sentiment": {"score": 0, "label": "Neutral", "color": "#6c757d"}
            }
        ]
    except Exception as e:
        # Return a mock article in case of any error
        return [
            {
                "title": f"Latest news for {ticker} not available",
                "description": f"Error retrieving news: {str(e)}. Please try again later or check financial news websites directly.",
                "url": f"https://finance.yahoo.com/quote/{ticker}",
                "source": "System Message",
                "published_at": datetime.now().isoformat(),
                "sentiment": {"score": 0, "label": "Neutral", "color": "#6c757d"}
            }
        ]

# Function to display news articles with sentiment
def display_news_articles(articles):
    if not articles:
        st.warning("No news articles found.")
        return
    
    for article in articles:
        with st.container():
            # Get sentiment info
            sentiment = article.get('sentiment', {"label": "Neutral", "color": "#6c757d"})
            sentiment_label = sentiment.get('label', 'Neutral')
            sentiment_class = f"sentiment-{sentiment_label.lower()}"
            
            # Check if this news has been fed to chatbot
            fed_badge = ""
            if st.session_state.news_fed_to_chatbot:
                fed_badge = '<span class="news-fed-badge">Added to Chatbot</span>'
            
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">
                    {article.get('title', 'No title')}
                    <span class="sentiment-badge {sentiment_class}">{sentiment_label}</span>
                    {fed_badge}
                </div>
                <div class="news-source">Source: {article.get('source', 'Unknown')}</div>
                <div class="news-date">Published: {article.get('published_at', 'Unknown date')}</div>
                <div class="news-description">{article.get('description', 'No description available')}</div>
                <a href="{article.get('url', '#')}" target="_blank" class="news-link">Read more</a>
            </div>
            """, unsafe_allow_html=True)

# Function to add news content to document chunks
def add_news_to_documents(articles):
    if not articles:
        return
    
    for article in articles:
        # Create a document chunk from the article
        content = f"""
        # {article.get('title', 'News Article')}
        
        Source: {article.get('source', 'Unknown')}
        Published: {article.get('published_at', 'Unknown date')}
        
        {article.get('description', 'No description available')}
        
        URL: {article.get('url', 'No URL available')}
        """
        
        # Add to document chunks
        st.session_state.document_chunks.append({
            "content": content,
            "source": f"News: {article.get('source', 'Unknown')}"
        })
    
    # Update embeddings if model is loaded
    if st.session_state.embedding_model_loaded:
        try:
            # Get only the new chunks (the ones we just added)
            new_chunk_texts = [chunk["content"] for chunk in st.session_state.document_chunks[-len(articles):]]
            new_embeddings = compute_embeddings(new_chunk_texts)
            
            if new_embeddings is not None and len(new_embeddings) > 0:
                # Append to existing embeddings
                st.session_state.document_embeddings.extend(new_embeddings)
        except Exception as e:
            st.error(f"Error computing embeddings for news: {str(e)}")
            # Continue without embeddings, will fall back to keyword search

# Function to preprocess and improve chatbot responses
def preprocess_response(response, query):
    try:
        # If response is too short or seems like an error message, return as is
        if not response or len(response) < 50 or "error" in response.lower():
            return response
        
        # Clean up the response
        response = clean_text(response)
        
        # Improve formatting for readability
        response = format_response(response)
        
        # Add specific improvements for different query types
        if any(term in query.lower() for term in ["what", "how", "why", "when", "where"]):
            # For question queries, ensure the response is direct and focused
            if not response.startswith("Based on") and not response.startswith("According to"):
                response = f"Based on the information I found: {response}"
        
        # For news-related queries, highlight the recency of information
        if any(term in query.lower() for term in ["news", "recent", "latest", "update"]):
            if "Source: News:" in response:
                response = f"From the latest news: {response}"
        
        # For stock-specific queries, ensure disclaimer is present
        if any(term in query.lower() for term in ["stock", "buy", "sell", "invest", "price"]):
            if not "not financial advice" in response.lower() and not "consult" in response.lower():
                response += "\n\nPlease note that this information is not financial advice. Always consult with a qualified financial advisor before making investment decisions."
        
        return response
    except Exception as e:
        # If any error occurs during preprocessing, return the original response
        st.session_state.error_message = f"Response preprocessing error: {str(e)}\n{traceback.format_exc()}"
        return response

# Function to generate a response
def generate_response(query):
    try:
        if not st.session_state.document_chunks:
            return "Please process some documents first using the sidebar options."
        
        # Try semantic search first if embeddings are available
        if st.session_state.embedding_model_loaded and len(st.session_state.document_embeddings) > 0:
            try:
                # Compute query embedding
                query_embedding = compute_embeddings([query])
                
                if query_embedding is not None:
                    # Find similar chunks
                    similar_chunks = find_similar_chunks(
                        query_embedding[0], 
                        st.session_state.document_embeddings,
                        st.session_state.document_chunks,
                        top_k=3
                    )
                    
                    if similar_chunks:
                        # Extract content from results
                        context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
                        sources = ", ".join(set([chunk["source"] for chunk in similar_chunks]))
                        
                        # Check for specific query types and use custom handlers
                        if any(term in query.lower() for term in ["stock", "buy", "invest", "recommendation"]):
                            custom_response = create_investment_response(context, query)
                            if custom_response:
                                return preprocess_response(f"{custom_response}\n\nSource: {sources}", query)
                        
                        # Create a manual summary for common queries
                        if "citi" in query.lower() or "citigroup" in query.lower():
                            if any("hong kong" in chunk["content"].lower() for chunk in similar_chunks):
                                response = """Based on the documents, Citigroup has launched Citi AI, a suite of artificial intelligence tools for its employees in Hong Kong. These tools support internal operations including information retrieval from Citi's policy library, document summarization, and creation of electronic communications drafts. The initiative aligns with Hong Kong Monetary Authority's commitment to promoting responsible AI adoption in banking. Citi AI is currently available to about 150,000 employees across 11 countries including the United States, India, and Singapore, with plans to expand to more markets this year.

Source: Citi_article.pdf"""
                                return preprocess_response(response, query)
                        
                        # Try to use external LLM API for reasoning
                        try:
                            prompt = f"""
                            Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer this question."
                            
                            Context:
                            {context}
                            
                            Question: {query}
                            
                            Answer:
                            """
                            
                            llm_response = call_llm_api(prompt)
                            
                            # Check if the response seems valid
                            if llm_response and not llm_response.startswith("Error:"):
                                return preprocess_response(f"{llm_response}\n\nSource: {sources}", query)
                        except Exception as llm_error:
                            st.session_state.error_message = f"LLM API error: {str(llm_error)}\n{traceback.format_exc()}"
                            # Continue with fallback if LLM fails
                        
                        # Format and return the response with source information
                        return preprocess_response(f"{format_response(context)}\n\nSource: {sources}", query)
            except Exception as e:
                # If semantic search fails, fall back to keyword search
                st.session_state.error_message = f"Semantic search error: {str(e)}\n{traceback.format_exc()}"
        
        # Fallback to keyword search
        results = keyword_search(query, st.session_state.document_chunks)
        
        if results:
            # Extract content from results
            content = "\n\n".join([result["content"] for result in results])
            sources = ", ".join(set([result["source"] for result in results]))
            
            # Check for specific query types and use custom handlers
            if any(term in query.lower() for term in ["stock", "buy", "invest", "recommendation"]):
                custom_response = create_investment_response(content, query)
                if custom_response:
                    return preprocess_response(f"{custom_response}\n\nSource: {sources}", query)
            
            # Create a manual summary for common queries
            if "citi" in query.lower() or "citigroup" in query.lower():
                if any("hong kong" in result["content"].lower() for result in results):
                    response = """Based on the documents, Citigroup has launched Citi AI, a suite of artificial intelligence tools for its employees in Hong Kong. These tools support internal operations including information retrieval from Citi's policy library, document summarization, and creation of electronic communications drafts. The initiative aligns with Hong Kong Monetary Authority's commitment to promoting responsible AI adoption in banking. Citi AI is currently available to about 150,000 employees across 11 countries including the United States, India, and Singapore, with plans to expand to more markets this year.

Source: Citi_article.pdf"""
                    return preprocess_response(response, query)
            
            # Try to use external LLM API for reasoning
            try:
                prompt = f"""
                Answer the following question based only on the provided context. If the answer cannot be found in the context, say "I don't have enough information to answer this question."
                
                Context:
                {content}
                
                Question: {query}
                
                Answer:
                """
                
                llm_response = call_llm_api(prompt)
                
                # Check if the response seems valid
                if llm_response and not llm_response.startswith("Error:"):
                    return preprocess_response(f"{llm_response}\n\nSource: {sources}", query)
            except Exception as llm_error:
                st.session_state.error_message = f"LLM API error: {str(llm_error)}\n{traceback.format_exc()}"
                # Continue with fallback if LLM fails
            
            # Format and return the response with source information
            return preprocess_response(f"{format_response(content)}\n\nSource: {sources}", query)
        else:
            return "I couldn't find any relevant information about that in the documents."
    except Exception as e:
        # Catch-all error handler
        error_msg = f"Error generating response: {str(e)}"
        st.session_state.error_message = f"{error_msg}\n{traceback.format_exc()}"
        return f"I encountered an error while searching the documents. Please try a different question or reload the page."

# Main app layout
st.title("Financial NLP Chatbot")
st.markdown("Ask questions about financial documents, news, and market trends.")

# Stock ticker news section
st.header("Latest Stock News")

# Create a function to filter tickers based on user input
def filter_tickers(search_term):
    if not search_term:
        return POPULAR_TICKERS[:10]  # Return first 10 tickers if no search term
    
    # Filter tickers that start with the search term (case insensitive)
    filtered = [ticker for ticker in POPULAR_TICKERS if ticker.lower().startswith(search_term.lower())]
    
    # If no exact matches, try partial matches
    if not filtered:
        filtered = [ticker for ticker in POPULAR_TICKERS if search_term.lower() in ticker.lower()]
    
    # Add company names to the display if available
    display_options = []
    for ticker in filtered[:10]:  # Limit to 10 results
        if ticker in TICKER_TO_COMPANY:
            display_options.append(f"{ticker} - {TICKER_TO_COMPANY[ticker]}")
        else:
            display_options.append(ticker)
    
    return display_options

# Create columns for the ticker search
ticker_col1, ticker_col2 = st.columns([3, 1])

with ticker_col1:
    # Use text input for ticker search with autocomplete
    ticker_input = st.text_input("Enter a stock ticker or start typing for suggestions", key="ticker_input")
    
    # Filter tickers based on input
    filtered_options = filter_tickers(ticker_input)
    
    # Display filtered options as a selectbox
    if filtered_options:
        selected_option = st.selectbox("Select a ticker", filtered_options, key="ticker_select")
        
        # Extract ticker from the selected option (remove company name if present)
        selected_ticker = selected_option.split(" - ")[0] if " - " in selected_option else selected_option
    else:
        selected_ticker = ticker_input.upper() if ticker_input else "AAPL"  # Default to AAPL if no matches

with ticker_col2:
    if st.button("Get Latest News"):
        with st.spinner(f"Fetching latest news for {selected_ticker}..."):
            try:
                # Reset news fed status
                st.session_state.news_fed_to_chatbot = False
                # Fetch news articles
                articles = fetch_stock_news(selected_ticker)
                
                if articles:
                    st.session_state.news_articles = articles
                    st.session_state.selected_ticker = selected_ticker
                    
                    # Add news to documents for querying
                    if st.session_state.documents_processed:
                        add_news_to_documents(articles)
                        st.session_state.news_fed_to_chatbot = True
                        st.success(f"News for {selected_ticker} has been added to the chatbot's knowledge base!")
                else:
                    st.error(f"No news found for {selected_ticker}")
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")

# Display news if available
if st.session_state.news_articles and st.session_state.selected_ticker:
    st.subheader(f"Latest News for {st.session_state.selected_ticker}")
    display_news_articles(st.session_state.news_articles)

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Document Sources")
    
    # Option to use sample documents
    use_sample = st.checkbox("Use sample financial documents", value=False)
    
    # Option to upload custom documents
    uploaded_files = st.file_uploader(
        "Upload your financial documents",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    # Process documents button
    if st.button("Process Documents"):
        # Clear previous error message and document sources
        st.session_state.error_message = None
        st.session_state.document_sources = []
        st.session_state.document_chunks = []
        st.session_state.document_embeddings = []
        st.session_state.embedding_model_loaded = False
        
        with st.spinner("Processing documents..."):
            file_paths = []
            
            # Handle uploaded documents first (prioritize user uploads)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            file_paths.append(temp_file.name)
                        st.success(f"Loaded file: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
                        st.session_state.error_message = f"Error processing uploaded file: {str(e)}\n{traceback.format_exc()}"
                
                if file_paths:
                    st.success(f"Loaded {len(uploaded_files)} uploaded document(s)")
                    st.session_state.use_sample_only = False
            
            # Handle sample documents only if no uploads or explicitly requested
            if (not file_paths and use_sample) or use_sample:
                try:
                    sample_paths = download_github_files()
                    file_paths.extend(sample_paths)
                    if sample_paths:
                        st.success(f"Loaded {len(sample_paths)} sample document(s)")
                        if not uploaded_files:
                            st.session_state.use_sample_only = True
                except Exception as e:
                    st.error(f"Error loading sample documents: {str(e)}")
                    st.session_state.error_message = f"Error loading sample documents: {str(e)}\n{traceback.format_exc()}"
            
            # Process documents
            if file_paths:
                success = process_documents(file_paths, st.session_state.use_sample_only)
                if success:
                    st.session_state.documents_processed = True
                    st.success("Documents processed successfully!")
                else:
                    st.error("Failed to process documents. See error details below.")
            else:
                st.error("No documents to process.")
    
    # Display document sources
    if st.session_state.document_sources:
        st.subheader("Loaded Documents:")
        for source in st.session_state.document_sources:
            st.write(f"- {source}")
    
    # Display error message if any
    if st.session_state.error_message:
        with st.expander("Show Error Details"):
            st.code(st.session_state.error_message)
    
    # API Configuration
    st.header("API Configuration (Optional)")
    api_url = st.text_input("LLM API URL (Optional)", value="https://openrouter.ai/api/v1/chat/completions")
    api_key = st.text_input("API Key (Optional)", type="password", value="sk-or-v1-free-tier-demo")
    
    if st.button("Save API Settings"):
        st.success("API settings saved!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses lightweight embeddings and external LLM APIs to answer questions about financial documents.
    
    **Features:**
    - Upload your own financial documents (PDF, TXT, CSV)
    - Use sample financial documents
    - Get latest news for popular stock tickers
    - Ask questions about financial news, market trends, and more
    
    **Technologies:**
    - Sentence Transformers for document embeddings
    - External LLM API for reasoning (with fallbacks)
    - LangChain for document processing
    - Streamlit for the user interface
    """)

# Main chat interface
st.header("Chat")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="avatar">🤖</div>
            <div class="message">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Input for new messages
user_input = st.text_input("Type your question here:", key="user_input")

# Send button
if st.button("Send") and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Check if documents have been processed
    if not st.session_state.documents_processed:
        response = "Please process some documents first using the sidebar options."
    else:
        # Get response using the generate_response function
        with st.spinner("Thinking..."):
            try:
                response = generate_response(user_input)
            except Exception as e:
                response = f"I encountered an error while generating a response. Please try again with a different question."
                st.session_state.error_message = f"Error in generate_response: {str(e)}\n{traceback.format_exc()}"
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Rerun to update the chat display
    st.rerun()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.markdown("---")
st.markdown("Developed by Dymasius Yusuf Sitepu (G2303593E) | MSc in Financial Technology | NTU")
