

import streamlit as st
import os
import json
from docx import Document
from datetime import datetime
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


from PyPDF2 import PdfReader





# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Constants
SIMILARITY_THRESHOLD = 0.85
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class ProjectManager:
    def __init__(self):
        self.base_path = "projects"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        os.makedirs(self.base_path, exist_ok=True)

    def create_project_structure(self, project_name):
        project_path = os.path.join(self.base_path, project_name)
        os.makedirs(os.path.join(project_path, "docs"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "vectors"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "chats"), exist_ok=True)
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "doc_count": 0,
            "chat_count": 0
        }
        self.save_metadata(project_name, metadata)

    def save_metadata(self, project_name, metadata):
        metadata_path = os.path.join(self.base_path, project_name, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def extract_text_from_file(self, file, file_type):
        try:
            text = ""
            if file_type == "pdf":
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file_type == "docx":
                doc = Document(file)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif file_type == "txt":
                text = file.read().decode("utf-8")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {file.name}: {str(e)}")
            return ""


    def process_document(self, file, project_name):
        try:
            file_type = file.name.split('.')[-1].lower()
            if file_type not in ['pdf', 'docx', 'txt']:
                st.error(f"Unsupported file type: {file_type}")
                return False

            doc_path = os.path.join(self.base_path, project_name, "docs", file.name)
            with open(doc_path, "wb") as f:
                f.write(file.getvalue())

            text = self.extract_text_from_file(file, file_type)
            if not text:
                st.warning(f"No text could be extracted from {file.name}")
                return False

            chunks = self.get_text_chunks(text)
            if not chunks:
                st.warning(f"No valid text chunks extracted from {file.name}")
                return False

            vector_store = FAISS.from_texts(chunks, self.embeddings)
            vector_store.save_local(
                os.path.join(self.base_path, project_name, "vectors", f"{file.name}_index")
            )
            return True

        except Exception as e:
            logger.error(f"Error processing document {file.name}: {str(e)}")
            return False

    def get_text_chunks(self, text):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            chunks = splitter.split_text(text)
            return [chunk for chunk in chunks if chunk.strip()]
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            return []

class ChatManager:
    def __init__(self, project_name):
        self.project_name = project_name
        self.chats_path = f"projects/{project_name}/chats"
        os.makedirs(self.chats_path, exist_ok=True)

    def create_chat(self, chat_name):
        chat_dir = os.path.join(self.chats_path, chat_name)
        os.makedirs(chat_dir, exist_ok=True)
        self.save_chat_metadata(chat_name, {
            "created_at": datetime.now().isoformat(),
            "messages": []
        })

    def save_chat_metadata(self, chat_name, metadata):
        metadata_path = os.path.join(self.chats_path, chat_name, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def load_chat_history(self, chat_name):
        chat_file = os.path.join(self.chats_path, chat_name, "history.json")
        if os.path.exists(chat_file):
            with open(chat_file, "r") as f:
                return json.load(f)
        return []

    def save_message(self, chat_name, message):
        history = self.load_chat_history(chat_name)
        history.append({
            "timestamp": datetime.now().isoformat(),
            "role": message["role"],
            "content": message["content"]
        })
        chat_file = os.path.join(self.chats_path, chat_name, "history.json")
        with open(chat_file, "w") as f:
            json.dump(history, f, indent=4)

    def get_chat_names(self):
        return [d for d in os.listdir(self.chats_path) if os.path.isdir(os.path.join(self.chats_path, d))]
class QASystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    def get_answer_from_doc(self, project_name, question, doc_name):
        try:
            vector_store_path = os.path.join("projects", project_name, "vectors", f"{doc_name}_index")
            if not os.path.exists(vector_store_path):
                return f"No index found for {doc_name}"

            vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            docs = vector_store.similarity_search(question, k=4)

            prompt_template = """
            Answer the question as detailed as possible from the provided context. If the answer is not in
            the context, just say "No relevant information found in this document."

            Context: {context}
            Question: {question}

            Answer:
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(
                self.model,
                chain_type="stuff",
                prompt=prompt
            )

            response = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )

            return response["output_text"]

        except Exception as e:
            logger.error(f"Error getting answer from {doc_name}: {str(e)}")
            return f"Error processing {doc_name}"

    def find_similar_question(self, question, chat_history):
        try:
            question_embedding = self.embeddings.embed_query(question)
            for entry in chat_history:
                if entry["role"] == "user":
                    prev_embedding = self.embeddings.embed_query(entry["content"])
                    similarity = cosine_similarity(
                        [question_embedding],
                        [prev_embedding]
                    )[0][0]
                    if similarity > SIMILARITY_THRESHOLD:
                        # Return the next assistant message
                        next_idx = chat_history.index(entry) + 1
                        if next_idx < len(chat_history) and chat_history[next_idx]["role"] == "assistant":
                            return chat_history[next_idx]["content"]
            return None
        except Exception as e:
            logger.error(f"Error finding similar questions: {str(e)}")
            return None


def display_chat_message(message, is_user=True):
    user_icon = "https://img.icons8.com/color/48/000000/user-male-circle--v1.png"
    ai_icon = "https://img.icons8.com/color/48/000000/artificial-intelligence.png"
    
    with st.container():
        col1, col2 = st.columns([6, 1]) if is_user else st.columns([1, 6])
        
        if is_user:
            with col1:
                st.markdown(f"""
                    <div style='background: linear-gradient(to right, #E0EAFC, #CFDEF3);
                                color: #333;
                                padding: 15px; 
                                border-radius: 15px 15px 15px 0;
                                margin: 5px;
                                text-align: left;
                                position: relative;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
                        <img src='{user_icon}' style='width: 30px; 
                                                    height: 30px;
                                                    border-radius: 50%; 
                                                    position: absolute;
                                                    top: -15px;
                                                    left: -15px;
                                                    border: 2px solid white;'>
                        <div style='margin-left: 20px;'>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"""
                    <div style='background: linear-gradient(to right, #8E2DE2, #4A00E0);
                                color: white;
                                padding: 15px; 
                                border-radius: 15px 15px 0 15px;
                                margin: 5px;
                                text-align: left;
                                position: relative;
                                box-shadow: -2px 2px 5px rgba(0,0,0,0.1);'>
                        <img src='{ai_icon}' style='width: 30px; 
                                                   height: 30px;
                                                   border-radius: 50%; 
                                                   position: absolute;
                                                   top: -15px;
                                                   right: -15px;
                                                   border: 2px solid white;'>
                        <div style='margin-right: 20px;'>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    
    project_manager = ProjectManager()
    qa_system = QASystem()

    with st.sidebar:
        st.title("📚 Document Chat System")
        st.header("Project Management")
        
        new_project = st.text_input("Create New Project")
        if st.button("Create Project"):
            if new_project:
                project_manager.create_project_structure(new_project)
                st.success(f"Project '{new_project}' created!")
                st.session_state.current_project = new_project
                st.session_state.messages = []

        projects = [d for d in os.listdir("projects") if os.path.isdir(os.path.join("projects", d))]
        if projects:
            selected_project = st.selectbox(
                "Select Project",
                projects,
                index=projects.index(st.session_state.current_project) if st.session_state.current_project in projects else 0
            )
            
            if selected_project != st.session_state.current_project:
                st.session_state.current_project = selected_project
                st.session_state.current_chat = None
                st.session_state.messages = []

            if st.session_state.current_project:
                chat_manager = ChatManager(st.session_state.current_project)
                
                new_chat = st.text_input("Create New Chat")
                if st.button("Create Chat"):
                    if new_chat:
                        chat_manager.create_chat(new_chat)
                        st.success(f"Chat '{new_chat}' created!")
                        st.session_state.current_chat = new_chat
                        st.session_state.messages = []

                chats = chat_manager.get_chat_names()
                if chats:
                    selected_chat = st.selectbox(
                        "Select Chat",
                        chats,
                        index=chats.index(st.session_state.current_chat) if st.session_state.current_chat in chats else 0
                    )
                    if selected_chat != st.session_state.current_chat:
                        st.session_state.current_chat = selected_chat
                        st.session_state.messages = chat_manager.load_chat_history(selected_chat)

                with st.expander("📁 Upload Documents", expanded=False):
                    uploaded_files = st.file_uploader(
                        "Upload your documents",
                        accept_multiple_files=True,
                        type=['pdf', 'docx', 'txt']
                    )

                    if uploaded_files and st.button("Process Documents"):
                        with st.spinner("Processing documents..."):
                            for file in uploaded_files:
                                if project_manager.process_document(file, st.session_state.current_project):
                                    st.success(f"Processed {file.name}")
                                else:
                                    st.error(f"Failed to process {file.name}")

    if st.session_state.current_project and st.session_state.current_chat:
        st.header(f"Project: {st.session_state.current_project} | Chat: {st.session_state.current_chat}")

        chat_container = st.container()
        
        with st.container():
            st.markdown("<div style='padding: 1rem;'></div>", unsafe_allow_html=True)
            user_input = st.chat_input("Ask a question about your documents...")
            
            if user_input:
                new_user_message = {"role": "user", "content": user_input}
                st.session_state.messages.append(new_user_message)
                chat_manager.save_message(st.session_state.current_chat, new_user_message)

                similar_response = qa_system.find_similar_question(user_input, st.session_state.messages)
                if similar_response:
                    new_assistant_message = {
                        "role": "assistant", 
                        "content": f"Based on a similar previous question:\n{similar_response}"
                    }
                else:
                    # Get answers from each document
                    docs_path = os.path.join("projects", st.session_state.current_project, "docs")
                    all_responses = []
                    
                    if os.path.exists(docs_path):
                        for doc in os.listdir(docs_path):
                            answer = qa_system.get_answer_from_doc(
                                st.session_state.current_project,
                                user_input,
                                doc
                            )
                            if "No relevant information found" not in answer:
                                all_responses.append(f"\n\nFrom {doc}:\n{answer}")
                    
                    combined_response = "\n---".join(all_responses) if all_responses else "\nNo relevant information found in any document."
                    new_assistant_message = {"role": "assistant", "content": combined_response}
                
                st.session_state.messages.append(new_assistant_message)
                chat_manager.save_message(st.session_state.current_chat, new_assistant_message)

        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(message, is_user=(message["role"] == "user"))

        # Add spacing at the bottom
        st.markdown("<div style='padding: 5rem;'></div>", unsafe_allow_html=True)

    else:
        st.info("Please create or select a project and chat to begin.")




# Update the CSS in the load_css function
def load_css():
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: #ffffff;
        }
        .chat-message.assistant {
            background-color: #475063;
            color: #ffffff;
        }
        .chat-message .avatar {
            width: 20px;
            height: 20px;
            margin-right: 1rem;
            font-size: 1.2rem;
        }
        .chat-message .message {
            flex-grow: 1;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)




# Custom CSS for chat interface


if __name__ == "__main__":
    load_css()
    main()
