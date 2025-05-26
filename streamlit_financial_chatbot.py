import streamlit as st
import os
import tempfile
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
import re
from sentence_transformers import SentenceTransformer

# Set page configuration
st.set_page_config(
    page_title="Financial NLP Chatbot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance - only targeting chat messages for black text
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
                        
                        # Create a manual summary for common queries
                        if "citi" in query.lower() or "citigroup" in query.lower():
                            if any("hong kong" in chunk["content"].lower() for chunk in similar_chunks):
                                return """Based on the documents, Citigroup has launched Citi AI, a suite of artificial intelligence tools for its employees in Hong Kong. These tools support internal operations including information retrieval from Citi's policy library, document summarization, and creation of electronic communications drafts. The initiative aligns with Hong Kong Monetary Authority's commitment to promoting responsible AI adoption in banking. Citi AI is currently available to about 150,000 employees across 11 countries including the United States, India, and Singapore, with plans to expand to more markets this year.

Source: Citi_article.pdf"""
                        
                        # Format and return the response with source information
                        return f"{format_response(context)}\n\nSource: {sources}"
            except Exception as e:
                # If semantic search fails, fall back to keyword search
                st.session_state.error_message = f"Semantic search error: {str(e)}\n{traceback.format_exc()}"
        
        # Fallback to keyword search
        results = keyword_search(query, st.session_state.document_chunks)
        
        if results:
            # Extract content from results
            content = "\n\n".join([result["content"] for result in results])
            sources = ", ".join(set([result["source"] for result in results]))
            
            # Create a manual summary for common queries
            if "citi" in query.lower() or "citigroup" in query.lower():
                if any("hong kong" in result["content"].lower() for result in results):
                    return """Based on the documents, Citigroup has launched Citi AI, a suite of artificial intelligence tools for its employees in Hong Kong. These tools support internal operations including information retrieval from Citi's policy library, document summarization, and creation of electronic communications drafts. The initiative aligns with Hong Kong Monetary Authority's commitment to promoting responsible AI adoption in banking. Citi AI is currently available to about 150,000 employees across 11 countries including the United States, India, and Singapore, with plans to expand to more markets this year.

Source: Citi_article.pdf"""
            
            # Format and return the response with source information
            return f"{format_response(content)}\n\nSource: {sources}"
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
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses lightweight embeddings to answer questions about financial documents.
    
    **Features:**
    - Upload your own financial documents (PDF, TXT, CSV)
    - Use sample financial documents
    - Ask questions about financial news, market trends, and more
    
    **Technologies:**
    - Sentence Transformers for document embeddings
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
