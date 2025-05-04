import os
import openai
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
GPT_MODEL = "text-embedding-3-small"
PDF_TABLE = "pdf_embeddings"

def get_query_embedding(query: str) -> list:
    response = openai.embeddings.create(
        model=GPT_MODEL,
        input=query
    )
    return response.data[0].embedding

def query_similar_chunks(user_query: str, top_k: int = 5):
    query_embedding = get_query_embedding(user_query)
    vector_str = str(query_embedding).replace('[', '{').replace(']', '}')

    sql = f"""
        SELECT chunk_text
        FROM {PDF_TABLE}
        ORDER BY embedding <#> '{vector_str}'
        LIMIT {top_k};
    """

    response = supabase.postgrest.rpc('sql', {"q": sql}).execute()
    
    if response.data:
        return [row['chunk_text'] for row in response.data]
    else:
        return []

if __name__ == "__main__":
    query = "What are the warranty terms of product X?"
    results = query_similar_chunks(query, top_k=3)

    print("\nTop Matching Chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk[:300]}...\n")  
