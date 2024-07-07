import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Jawaban tidak tersedia dari konteks yang diberikan", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_response(response_text):
    questions = response_text.strip().split('\n\n')
    formatted_questions = []
    
    for question in questions:
        lines = question.strip().split('\n')
        formatted_question = []
        for line in lines:
            if line.strip():
                formatted_question.append(line.strip())
        formatted_questions.append('\n'.join(formatted_question))
    
    return '\n\n'.join(formatted_questions)

def answer_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.markdown(f"**Reply:**\n\n{response['output_text']}")

def format_quiz(raw_quiz):
    formatted_quiz = []
    questions = re.split(r'\d+\.', raw_quiz)[1:]  # Split by question numbers

    for i, question in enumerate(questions, 1):
        lines = question.strip().split('\n')
        question_text = lines[0].strip()
        
        # Check if the question is a placeholder and skip it if so
        if question_text.startswith('[Generated Question'):
            continue
        
        options = []
        answer = ""

        for line in lines[1:]:
            line = line.strip()
            if line.startswith(('A.', 'B.', 'C.', 'D.')):
                options.append(line)
            elif 'Answer:' in line:
                answer = line

        formatted_question = f"{i}. Question: {question_text}\n"
        for option in options:
            formatted_question += f"   {option}\n"
        formatted_question += f"   {answer}\n"

        formatted_quiz.append(formatted_question)

    return "\n".join(formatted_quiz)

def generate_quiz(context_language, quiz_language, num_questions):
    question_format = """
    {i}. [Write a specific, relevant question here based on the context]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Answer: [Correct Answer]
    """
    questions_prompt = "\n\n".join([question_format.format(i=i+1) for i in range(num_questions)])
    
    prompt_template = f"""
    Create a multiple-choice quiz in {quiz_language} based on the given context in {context_language}. Follow these rules strictly:
    1. Ensure each question is relevant and specific to the context provided.
    2. Provide 4 answer options (A, B, C, D) for each question.
    3. Indicate the correct answer after the options.
    4. Do not use placeholders like "[Write a specific, relevant question here based on the context]". Instead, write actual, context-specific questions.
    5. The output format should be exactly as shown below, with each question on a new line and options labeled A, B, C, D:

    Context:\n {{context}}\n

    Quiz:
    {questions_prompt}
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    # Get context from the uploaded PDF
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search("", k=5)  # Get top 5 most relevant chunks
    context = " ".join([doc.page_content for doc in docs])
    
    # Generate quiz
    response = chain(
        {"input_documents": docs, "question": prompt.format(context=context)},
        return_only_outputs=True
    )
    
    # Format the generated quiz
    formatted_quiz = format_quiz(response["output_text"])
    
    # Display the generated quiz
    st.write("Generated Quiz:")
    st.code(formatted_quiz)
    
def main():
    
    st.set_page_config(page_title="ByteBrain - Chat with PDF", page_icon="🤖")
    st.header("ByteBrain - Chat with PDF 📂")
    st.markdown("Fitur utama dari sistem ini adalah user dapat memperoleh informasi berdasarkan file PDF yang di upload dan dapat membuat daftar pertanyaan atau kuis beserta jawabannya. Silahkan upload file PDF Anda, form upload terletak pada pojok kiri atas Browser ↖️", unsafe_allow_html=True)

    quiz_language = st.selectbox("Pilih Bahasa Kuis:", ["Indonesian", "English"])
    num_questions = st.number_input("Masukkan jumlah soal yang diinginkan", min_value=1, max_value=100, value=10)

    if st.button("Generate Quiz"):
        if os.path.exists("faiss_index"):
            with st.spinner("Generating quiz..."):
                generate_quiz("Indonesian", quiz_language, num_questions)
        else:
            st.warning("Tolong upload tekan tombol Ekstrak Teks dari PDF terlebih dahulu")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload file PDF anda (bisa beberapa file) lalu klik tombol Ekstrak Teks dari PDF, setelah itu Anda bisa tanyakan pertanyaan atau generate quiz.", accept_multiple_files=True)
        if st.button("Ekstrak Teks dari PDF"):
            with st.spinner("Memuat..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Selesai")

        st.header("Ajukan Pertanyaan:")
        user_question = st.text_input("Masukkan pertanyaan Anda di sini:")
        if st.button("Kirim"):
            if user_question:
                if os.path.exists("faiss_index"):
                    with st.spinner("Mencari jawaban..."):
                        answer_question(user_question)
                else:
                    st.warning("Tolong upload tekan tombol Ekstrak Teks dari PDF terlebih dahulu")

if __name__ == "__main__":
    main()