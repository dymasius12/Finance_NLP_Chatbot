import streamlit as st
import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
import io
import zipfile
from pathlib import Path
import traceback

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
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'document_sources' not in st.session_state:
    st.session_state.document_sources = []
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Function to download sample financial documents from GitHub
@st.cache_data
def download_github_files():
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

# Function to process documents and create a retrieval QA chain
def process_documents(file_paths):
    try:
        documents = []
        
        # Load documents based on file type
        for file_path in file_paths:
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    st.session_state.document_sources.append(f"PDF: {os.path.basename(file_path)}")
                
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    st.session_state.document_sources.append(f"Text: {os.path.basename(file_path)}")
                
                elif file_extension == '.csv':
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())
                    st.session_state.document_sources.append(f"CSV: {os.path.basename(file_path)}")
                
                else:
                    st.warning(f"Unsupported file type: {file_extension}")
            
            except Exception as e:
                st.error(f"Error processing file {file_path}: {str(e)}")
                st.session_state.error_message = f"Error processing file: {str(e)}\n{traceback.format_exc()}"
        
        if not documents:
            st.error("No documents could be processed. Please check file formats and try again.")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            st.error("No text chunks were created. Documents may be empty or unreadable.")
            return None
        
        # Create embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.error(f"Error loading embeddings model: {str(e)}")
            st.session_state.error_message = f"Error loading embeddings model: {str(e)}\n{traceback.format_exc()}"
            return None
        
        # Create vector store using FAISS instead of Chroma
        try:
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.session_state.error_message = f"Error creating vector store: {str(e)}\n{traceback.format_exc()}"
            return None
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Get Hugging Face API token
        huggingface_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        
        if not huggingface_api_token:
            st.warning("⚠️ Hugging Face API token not found. The chatbot will retrieve documents but won't generate answers.")
            return retriever
        
        # Create language model
        try:
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",  # Using a more stable model
                huggingfacehub_api_token=huggingface_api_token,
                model_kwargs={"temperature": 0.5, "max_length": 512}
            )
        except Exception as e:
            st.error(f"Error loading language model: {str(e)}")
            st.session_state.error_message = f"Error loading language model: {str(e)}\n{traceback.format_exc()}"
            return retriever  # Return just the retriever if LLM fails
        
        # Create prompt template
        template = """
        You are a helpful financial assistant that provides information based on the documents given to you.
        Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        If you don't know the answer or can't find it in the context, just say "I don't have enough information to answer this question." Don't try to make up an answer.
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            return qa_chain
        except Exception as e:
            st.error(f"Error creating QA chain: {str(e)}")
            st.session_state.error_message = f"Error creating QA chain: {str(e)}\n{traceback.format_exc()}"
            return retriever  # Return just the retriever if chain creation fails
            
    except Exception as e:
        st.error(f"Unexpected error during document processing: {str(e)}")
        st.session_state.error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        return None

# Main app layout
st.title("Financial NLP Chatbot")
st.markdown("Ask questions about financial documents, news, and market trends.")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Document Sources")
    
    # Option to use sample documents
    use_sample = st.checkbox("Use sample financial documents", value=True)
    
    # Option to upload custom documents
    uploaded_files = st.file_uploader(
        "Upload your financial documents",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
    )
    
    # Process documents button
    if st.button("Process Documents"):
        # Clear previous error message
        st.session_state.error_message = None
        
        with st.spinner("Processing documents..."):
            file_paths = []
            
            # Handle sample documents
            if use_sample:
                try:
                    sample_paths = download_github_files()
                    file_paths.extend(sample_paths)
                    st.success(f"Loaded {len(sample_paths)} sample document(s)")
                except Exception as e:
                    st.error(f"Error loading sample documents: {str(e)}")
                    st.session_state.error_message = f"Error loading sample documents: {str(e)}\n{traceback.format_exc()}"
            
            # Handle uploaded documents
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
            
            # Process documents
            if file_paths:
                st.session_state.qa_chain = process_documents(file_paths)
                if st.session_state.qa_chain is not None:
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
    
    # API key configuration
    st.header("API Configuration")
    api_key = st.text_input("Hugging Face API Token", type="password", value=os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
    if st.button("Save API Key"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        st.success("API key saved!")
        # If documents are already processed, reprocess them with the new API key
        if st.session_state.documents_processed and st.session_state.document_sources:
            file_paths = download_github_files()
            st.session_state.qa_chain = process_documents(file_paths)
            st.success("Documents reprocessed with new API key!")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about financial documents.
    
    **Features:**
    - Upload your own financial documents (PDF, TXT, CSV)
    - Use sample financial documents
    - Ask questions about financial news, market trends, and more
    
    **Technologies:**
    - LangChain for document processing and retrieval
    - Hugging Face for embeddings and language model
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
    elif st.session_state.qa_chain is None:
        response = "I'm having trouble accessing the language model. Please check your API key configuration."
    else:
        # Get response from QA chain
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(user_input)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            st.session_state.error_message = f"Error generating response: {str(e)}\n{traceback.format_exc()}"
    
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
