# Document Chat System 📚💬
An intelligent, multi-project document chat system powered by Google's Generative AI and Streamlit. This application allows users to create multiple projects, upload various document types, and engage in AI-assisted conversations based on the content of those documents.

## ✨ Features
- **Multi-Project Management**: Create and switch between multiple projects.
- **Document Processing**:
  - Support for PDF, DOCX, and TXT files.
  - Efficient text extraction and chunking.
- **Intelligent Chatbot**:
  - Document-specific answers.
  - Similar question detection for faster responses.
  - Context-aware AI interactions.
- **Multiple Chat Sessions**: Create and manage different chat conversations within each project.
- **Modern UI**:
  - Sleek, gradient-style chat bubbles.
  - Distinct user and AI icons.
  - Fixed chat input for better UX.
- **Persistent Storage**: Saves projects, documents, and chat histories.

## 🚀 Quick Start
### Clone the repository
```bash
git clone <repository_url>
cd advanced-document-chat
```
### Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Set up your Google API Key
Create a `.env` file or modify an existing one in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```
### Run the application
```bash
streamlit run app.py
```

## 📚 How to Use
### Create a New Project
- Use the sidebar to create a new project.
- Enter a unique name for your project.

### Upload Documents
- Select your project from the sidebar.
- Use the "Upload Documents" expander to add PDF, DOCX, or TXT files.
- Click "Process Documents" to extract and index the content.

### Create a Chat Session
- Use the sidebar to create a new chat within your project.
- Enter a name for your chat session.

### Start Chatting
- Select a chat session.
- Type your questions in the chat input at the bottom.
- Receive AI-generated answers based on your uploaded documents.

## 🏢 Project Structure
```
advanced-document-chat/
├── streamlit_app.py
├── requirements.txt
├── .env
```

## 🛠️ Technical Details
- **Frontend**: Streamlit
- **AI Model**: Google's Generative AI (Gemini Pro)
- **Embeddings**: Google Generative AI Embeddings
- **Vector Store**: FAISS for efficient similarity search
- **PDF Processing**: PyPDF2
- **DOCX Processing**: python-docx
- **Text Chunking**: LangChain's RecursiveCharacterTextSplitter

## ⚙️ Configuration
- **Chunk Size**: 1000 characters (adjustable in `CHUNK_SIZE` constant)
- **Chunk Overlap**: 200 characters (adjustable in `CHUNK_OVERLAP` constant)
- **Similarity Threshold**: 0.85 for similar question detection (adjustable in `SIMILARITY_THRESHOLD` constant)

## 🔍 Troubleshooting
- **PDF Extraction Issues**: Ensure PDFs are not encrypted and contain extractable text.
- **API Key Errors**: Verify your Google API key in the `.env` file and check your Google Cloud Console for proper permissions.
- **Memory Issues**: For large documents, consider adjusting the `CHUNK_SIZE` or processing fewer documents simultaneously.

---

# Retrieval Non Augmented Generation (RNAG)
## Overview
Retrieval Non Augmented Generation (RNAG) is an innovative approach in natural language processing that refrains from conventional data augmentation methods used in many generative models. Instead, RNAG focuses on directly answering queries based solely on internal data, thus eliminating the need for incorporating external information sources like retrieval-augmented generation (RAG) techniques.

## ✨ Key Features
- **Non-Augmentation**: Unlike RAG variants that depend on data augmentation, RNAG leverages internal logical constructs to make inferences and generate responses.
- **Direct Query Handling**: The approach utilizes input queries as they are, fostering quicker response times by sidestepping complex retrieval processes.
- **Session Management**: Organizes dialogue through defined sessions, maintaining state (queries, responses) across interactions.
- **Embedding and Similarity**: Employs embedding techniques to assess similarity between queries and stored contexts, ensuring relevant responses.

## 📁 Implementation Details
The core implementation of RNAG as demonstrated in the provided code showcases handling incoming queries through various functions. Here are a few key components:

### JsonDB Class
Handles the storage and retrieval of queries and their corresponding responses using embeddings for similarity comparison.

### Session Class
Manages individual user sessions, storing conversations and handling offline queries.

### MainProcessWorkflow Class
Coordinates the overall workflow, managing sessions, handling queries, and processing offline queries.

## 🔍 Key Methods
- `start_new_session`: Initializes a new session.
- `handle_query`: Processes incoming queries, retrieves or generates responses based on session data.
- `offline_cosine_similarity`: Computes similarity between queries and stored data when offline.
- `process_offline_queries`: Processes stored offline queries when back online.
- `save_session_to_json`: Saves session data to a JSON file.
- `load_session_from_json`: Loads session data from a JSON file.
- `load_all_sessions`: Loads all session data from the session folder.
- `clear_memory`: Clears unused variables to free up memory.
- `reinitialize_models`: Reinitializes models to ensure they are up-to-date.

## 🔧 Setup and Installation
### Prerequisites
- Python 3.7+
- Hugging Face Token

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set the Hugging Face Token:**
   ```bash
   export HUGGINGFACE_TOKEN=<your_hugging_face_token>
   ```
4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 🚀 Running in Google Colab
### Install LocalTunnel:
```bash
!npm install localtunnel
```
### Get IP Address for LocalTunnel:
```python
import urllib
print("Password/Endpoint IP for LocalTunnel is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
```
### Run Streamlit and LocalTunnel:
```bash
!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501
```
### Access the application:
A link will be displayed. Click it and paste the IP address of LocalTunnel in it. Let the code run.

---

This README provides a complete guide for setting up and using the Document Chat System and RNAG. Let us know if you need any further modifications! 🚀

