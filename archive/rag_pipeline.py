from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import get_config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_pipeline():
    """Load and configure the RAG pipeline using centralized configuration."""
    config = get_config()
    logger.info(f"Initializing RAG pipeline with model: {config.model_name}")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    
    # Load vectorstore
    vectorstore = FAISS.load_local(
        config.vectorstore_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.retrieval_k})

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.model_name, 
        temperature=config.temperature, 
        max_tokens=config.max_tokens
    )

    # Create prompt template
    rag_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""You are an expert assistant for education policy question-answering tasks. 
Use the following pieces of retrieved context to answer the question accurately and comprehensively.

Guidelines:
- Provide evidence-based answers using the provided context
- If you don't know the answer, clearly state that you don't know
- Keep answers concise but informative (aim for 2-4 sentences)
- Cite specific information from the context when possible
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:        
    """
    )

    # Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    logger.info("RAG pipeline initialized successfully")
    return rag_chain

def ask_question(question):
    """Ask a question using the RAG pipeline."""
    logger.info(f"Processing question: {question}")
    rag_chain = load_qa_pipeline()
    response = rag_chain.invoke({"input": question})
    logger.info("Question processed successfully")
    return response

if __name__ == "__main__":
    query = "What are the latest trends in education policy?"
    answer = ask_question(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")