import os
import fitz  
import openai
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200,  
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    print(f"\n📄 Processing: {pdf_name}")

    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(full_text)
    print(f"🧠 Chunks created: {len(chunks)}")

    for i, chunk in enumerate(tqdm(chunks, desc="Uploading chunks")):
        embedding = get_embedding(chunk)
        data = {
            "pdf_name": pdf_name,
            "chunk_text": chunk,
            "embedding": embedding
        }
        supabase.table("pdf_embeddings").insert(data).execute()
    print(f"✅ Uploaded all chunks for: {pdf_name}")

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            process_pdf(file_path)

if __name__ == "__main__":
    folder_path = r"C:\Users\sv47154\Desktop\Chatbot\large_docs"  
    process_folder(folder_path)
