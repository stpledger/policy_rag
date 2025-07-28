import streamlit as st
from rag_pipeline import ask_question

st.title("🏛️💬 Education Policy Q&A")

# UI
question = st.text_input("Ask a question about education policy:")

if question:
    with st.spinner("Thinking..."):
        result = ask_question(question)
        st.subheader("Answer")
        st.write(result.get('answer'))

        sources = result.get('context')
        if sources:
            st.subheader("Sources")
            seen_urls = set()
            unique_sources = []
            for doc in sources:
                url = doc.metadata.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(doc)
            for doc in unique_sources:
                title = doc.metadata.get('title')
                url = doc.metadata.get('url')
                st.write(f"• [{title}]({url}), published on {doc.metadata.get('date')}, by {doc.metadata.get('authors')}")