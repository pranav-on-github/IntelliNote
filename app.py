from dotenv import load_dotenv
import streamlit as st
import webbrowser
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, ValidationError

class PDFInput(BaseModel):
    content: str

def main():
    st.set_page_config(page_title="IntelliNote")
    st.header("IntelliNote")

    st.write("Get your notes from Google Drive by clicking the button below:")
    url = "https://drive.google.com/drive/folders/1Lk59jzlR1PDM7pfO8tJEiLPUB8_vfn6C"
    if st.button('Open Drive'):
        webbrowser.open_new_tab(url)
    pdf = st.file_uploader("Upload your PDF to ask questions", type="pdf")

    if pdf is not None:
        text = pdf.read().decode("utf-8")
        try:
            input_data = PDFInput(content=text)
        except ValidationError as e:
            st.error(f"Validation error: {e.json()}")
            return

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(input_data.content)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
