from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv(dotenv_path=r'E:\Projects\Unfinished\Medical Chatbot\.env.local')

api_key_pinecone = os.getenv('API_KEY_PINECONE')
os.environ["PINECONE_API_KEY"] = api_key_pinecone

extracted_data = load_pdf(r'E:\Projects\Unfinished\Medical Chatbot\Data')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=api_key_pinecone)

index_name = "medbot"
print(f"Index Name: ", index_name)
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)



