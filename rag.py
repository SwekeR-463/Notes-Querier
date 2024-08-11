import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
import textwrap

#The models used
embed_model = "nomic-embed-text"
model = "gemma2:2b"

#The Prompt Template
template = """
### System:
You are a teaching assistant. Answer the user's questions using on the context provided. \
Your responses should be clear, concise, and structured in a way that helps the user understand and \
retain the information for their engineering semester exams. \
Add extra information too as per your training knowledge.

### Context:
{context}

### User:
{question}

### Response:
"""

prompt = PromptTemplate.from_template(template)

#Function for the Retrieval QA Chain
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        chain_type="stuff",
        return_source_documents=True, 
        chain_type_kwargs={'prompt': prompt} 
    )

#Function for getting the response
def get_response(query, chain):
    response = chain.invoke({'query': query})
    wrapped_text = textwrap.fill(response['result'], width=100)
    return wrapped_text

def process_pdf_and_answer_query(pdf, query):
    # Loading the PDF
    loader = PyMuPDFLoader(pdf.name)
    docs = loader.load()

    # Spliting the documents into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=0
    )
    texts = text_splitter.split_documents(docs)

    # Generating embeddings
    embeddings = OllamaEmbeddings(model=embed_model)

    # Loading the LLM from Ollama
    llm = Ollama(model=model)

    # Creating FAISS index
    db = FAISS.from_documents(texts, embeddings)

    # Creating retriever
    retriever = db.as_retriever()

    # Loading the QA chain
    chain = load_qa_chain(retriever=retriever, llm=llm, prompt=prompt)

    # Getting response
    return get_response(query, chain)


def gradio_interface(pdf, query):
    return process_pdf_and_answer_query(pdf, query)

#Interface using gradio
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Enter your question")
    ],
    outputs="text",
    title="Notes Querier",
    description="Upload your PDF of notes and ask any questions related to the topics in your notes.",
)


interface.launch()