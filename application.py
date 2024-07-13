import streamlit as st
from dotenv import load_dotenv
from htmlTemplate import css
from util import *

def main():
    load_dotenv()
    st.set_page_config(page_title="אופיס לייט")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("אופיס לייט  ")
    st.subheader("Upload  multiple PDFs and ask questions based on the PDF content.")
    st.text("* If you are using a mobile device, upload pdf by clicking on the top left arrow ")
    user_question = st.text_input("תוכל לשאול שאלות כאן")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload PDF/PDFs")
        pdf_docs = st.file_uploader("Upload and click on 'Process' (You can upload multiple PDFs)", 
                                    accept_multiple_files=True)
        if st.button("העלאה"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

            st.success('Ready to query', icon="✅")
if __name__ == '__main__':
    main()
