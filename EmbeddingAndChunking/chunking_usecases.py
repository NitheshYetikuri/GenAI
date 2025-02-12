
#pip install langchain_community python-dotenv langchain_chroma langchainhub --upgrade langchain_experimental langchain -q langchain_experimental langchain-google-genai tiktoken pypdf

# Import statements
import re
import os
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
api_key = ""
os.environ["GOOGLE_API_KEY"] = api_key

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = PyPDFLoader("/content/DocumentChunking_ResearchPaper.pdf")
data = loader.load()[0].page_content
headers_to_split_on = [("•", "Title"), ("→", "Sub title")]
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
document_chunks = text_splitter.split_text(data)
document_chunks

document_vectorestore=Chroma.from_documents(documents=document_chunks, embedding=embeddings)

retriever = document_vectorestore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("provide results?")
retrieved_docs



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3, max_tokens=500)



system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "Tell me about results of Medical Research Paper?"})
print("RAG Output:", response["answer"])

import uuid

# Define a Document class with page_content, metadata, and id attributes
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = str(uuid.uuid4())  # Generate a unique ID for each document

# Fixed size chunking
loader = PyPDFLoader("/content/FIxed_size_newsArticle.pdf")
data = loader.load()[0].page_content
text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
fixed_chunks = text_splitter.split_text(data)
print("Fixed size Chunking:", fixed_chunks)
documents_fixed = [Document(page_content=chunk) for chunk in fixed_chunks]

fixed_vectorstore=Chroma.from_documents(documents=documents_fixed, embedding=embeddings)

retriever = fixed_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("New policies?")
retrieved_docs[0].page_content

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "Tell me about new about Politics?"})
print("RAG Output for fixed size chunking:", response["answer"])

import uuid
from langchain_chroma import Chroma

# Define a Document class with page_content, metadata, and id attributes
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = str(uuid.uuid4())  # Generate a unique ID for each document

# Load the PDF content
loader = PyPDFLoader("/content/Semantic_CustomerSupport.pdf")
data = loader.load()[0].page_content

# Perform semantic chunking
chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount=20.0)
semantic_chunks = chunker.split_text(data)
print("Semantic Chunking:", semantic_chunks)

# Convert the chunks into Document objects
documents = [Document(page_content=chunk) for chunk in semantic_chunks]

semantic_vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

retriever = semantic_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("Lost password?")
retrieved_docs[0].page_content

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "I lost my password to do next?"})
print("RAG Output for Semantic chunking:", response["answer"])

import uuid


# Define a Document class with page_content, metadata, and id attributes
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = str(uuid.uuid4())  # Generate a unique ID for each document
# Recursive character splitting
loader = PyPDFLoader("/content/Recursive_LegalDocument.pdf")
data = loader.load()[0].page_content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=['\n', '. ', ' '])
recursive_chunks = text_splitter.split_text(data)
print("Recursive Chunking:", recursive_chunks)
documents_recursive = [Document(page_content=chunk) for chunk in recursive_chunks]

recursive_vectorstore=Chroma.from_documents(documents=documents_recursive, embedding=embeddings)

retriever = recursive_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("What is the contract period?")
retrieved_docs[0].page_content

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "Whats the duration of contract and what someone not follow that?"})
print("RAG Output for Recursive chunking:", response["answer"])

import uuid

# Define a Document class with page_content, metadata, and id attributes
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata
        self.id = str(uuid.uuid4())  # Generate a unique ID for each document

# Sentence chunking
loader = PyPDFLoader("/content/sentence_FAQ.pdf")
data = loader.load()[0].page_content
def sentence_chunking(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences
sentence_chunks = sentence_chunking(data)
print("Sentence Chunking:", sentence_chunks)
documents_sentence = [Document(page_content=chunk) for chunk in sentence_chunks]

sentence_vectorstore=Chroma.from_documents(documents=documents_sentence, embedding=embeddings)

retriever = sentence_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("where is my order?")
retrieved_docs[0].page_content

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "i want to know where is my order?"})
print("RAG Output for sentence chunking:", response["answer"])