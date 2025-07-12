import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from google.cloud import secretmanager

def get_openai_key():
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/rag-finance-demo/secrets/openai-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def load_qa_pipeline():

    openai_api_key = get_openai_key()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.load_local("faiss_finance_news", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_tokens=1000, openai_api_key=openai_api_key)

    rag_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:        
    """
    )

    combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain

def ask_question(question):
    rag_chain = load_qa_pipeline()
    response = rag_chain.invoke({"input": question})
    return response

if __name__ == "__main__":
    query = "What are the latest trends in the stock market?"
    answer = ask_question(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")