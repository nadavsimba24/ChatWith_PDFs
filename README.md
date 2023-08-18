# Welcome to Chat with PDFs

Chat with PDFs is a Python web application hosted on Streamlit, designed to empower users to seamlessly upload multiple PDFs and engage in insightful interactions through a conversational interface that revolves around the content of these PDFs.

## Give It a Try

🔗 To experience the capabilities of Chat with PDFs firsthand, simply navigate to [this link](https://chatwithpdfs.streamlit.app/) and dive into the interactive world of PDF-based conversations.

## Libraries Utilized

The Chat with PDFs project extensively employs the following key libraries:

- **Streamlit**: The web application is built upon the Streamlit framework, facilitating a user-friendly interface.
- **PyPDF2**: PDF content extraction is made possible by PyPDF2's `PdfReader`, enhancing text processing capabilities.
- **HuggingFace**: The `BAAI/bge-small-en` model from HuggingFace is instrumental in embedding text segments and forming a robust vector store.
- **Langchain and ChatOpenAI**: These components leverage the vector store to craft dynamic conversation chains, enabling engaging interactions.

## Workflow

The workflow of Chat with PDFs is as follows:

1. The Streamlit-based web application is both developed and hosted, serving as the entry point for users to submit one or multiple PDF files.
2. Utilizing PyPDF2's `PdfReader`, the application extracts textual content from the uploaded PDFs, subsequently forwarding it to the `CharacterTextSplitter` component.
3. `CharacterTextSplitter` takes charge of segmenting the extracted text into smaller, more manageable chunks, enhancing the subsequent processing steps.
4. The distinguished `BAAI/bge-small-en` model from HuggingFace is harnessed to embed the segmented text chunks, resulting in the creation of a comprehensive vector store.
5. This vector store proves pivotal in the construction of dynamic conversation chains through the combined utilization of Langchain and ChatOpenAI.

Thank you for your interest in Chat with PDFs.
