import streamlit as st
from utils import *
from dotenv import load_dotenv
from template import css, bot_template, user_template

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header('Chat with PDFs')

    user_input= st.text_input("Ask a question about the document you uploaded")
    if user_input:
        response = st.session_state.conversation({'question': user_input})
        st.session_state.chat_history = response['chat_history']
        st.write(response['answer'])
        
        # for i, response in enumerate(st.session_state.chat_history):
        #     if i % 2 == 0:
        #         st.write(user_template.replace(
        #             "{{MSG}}", response['question']), unsafe_allow_html=True)
        #     else:
        #         st.write(bot_template.replace(
        #             "{{MSG}}", response['answer']), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your Document")
        pdf_doc= st.file_uploader("Upload your PDF", accept_multiple_files=True)
        if st.button("Upload and Process"):
            with st.spinner("Processing..."):
        
                # get pdf text
                raw_text = get_text(pdf_doc=pdf_doc)
                
                # create chunks
                chunks = get_chunks(text=raw_text)

                # vector store
                vector_store = get_vector_store(text_chunks= chunks)

                # create converstional chain
                st.session_state.conversation = get_converse_chain(vectorstore=vector_store)

            st.success('Ready to query', icon="✅")

if __name__ == '__main__':
    main()

