import streamlit as st
from rag_pipeline import ask_question

st.title("📈💬 Finance News Q&A")

# UI
question = st.text_input("Ask a question about the latest finance news:")

if question:
    with st.spinner("Thinking..."):
        result = ask_question(question)
        st.subheader("Answer")
        # Handle both dict and object result types
        if isinstance(result, dict):
            st.write(result.get('answer') or result.get('result'))
            sources = result.get('context') or result.get('source_documents')
            if sources:
                st.subheader("Sources")
                for doc in sources:
                    title = doc.metadata.get('title') if hasattr(doc, 'metadata') else doc.get('title')
                    url = doc.metadata.get('url') if hasattr(doc, 'metadata') else doc.get('url')
                    st.write(f"• [{title}]({url})")
        else:
            st.write(result)