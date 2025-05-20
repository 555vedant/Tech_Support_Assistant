

import pandas as pd
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChatBot:
    def __init__(self):
        # load environment variables
        load_dotenv()

        # load the csv and dataset
        self.df = self.load_dataset('data/tech_support_vedant.csv')
        if self.df.empty:
            raise ValueError("Failed to load dataset. Ensure tech_support_dataset.csv exists.")

        documents = [
            Document(
                page_content=f"Issue: {row['Customer_Issue']}\nResponse: {row['Tech_Response']}",
                metadata={"category": row['Issue_Category']}
            )
            for _, row in self.df.iterrows()
        ]

        # initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            # Embed dataset issues for static fallback
            self.issue_embeddings = np.array([self.embeddings.embed_query(row['Customer_Issue']) for _, row in self.df.iterrows()])
        except Exception as e:
            print(f"Embedding initialization failed: {e}")
            self.embeddings = None
            self.issue_embeddings = None

        # initialize Pinecone
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = "tech-support-demo"
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=384,  # Matches all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            self.docsearch = PineconeVectorStore.from_documents(
                documents, self.embeddings, index_name=index_name
            ) if self.embeddings else None
        except Exception as e:
            print(f"Pinecone initialization failed: {e}")
            self.docsearch = None

        # initialize LLM
        self.llm = self.initialize_llm()

        #template 
        template = """
        You are a technical support assistant for TaskFlow, a cloud-based project management tool.
        Based on the provided context, answer the user's technical query clearly and concisely in no more than 2 sentences.
        If the context is insufficient, provide a general response or say you don't know.

        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # RetrievalQA chain
        self.rag_chain = None
        if self.llm and self.docsearch:
            try:
                self.rag_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.docsearch.as_retriever(search_kwargs={"k": 5}),  # Increased k for better retrieval
                    chain_type_kwargs={"prompt": self.prompt},
                    return_source_documents=True
                )
            except Exception as e:
                print(f"RetrievalQA initialization failed: {e}")

    def load_dataset(self, file_path):
        """
        load and process the CSV dataset.
        """
        try:
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['Customer_Issue', 'Tech_Response'])
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def initialize_llm(self):
        """
        initialize LLM with HuggingFace, fallback to Mixtral if Flan-T5 fails.
        """
        hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not hf_api_key:
            print("HUGGINGFACEHUB_API_TOKEN not found in .env file.")
            return None
        try:
            return HuggingFaceEndpoint(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=hf_api_key,
                temperature=0.7
            )
        except Exception as e:
            print(f"Flan-T5 initialization failed: {e}, trying Mixtral")
            try:
                return HuggingFaceEndpoint(
                    repo_id="mistral/mixtral-8x7b-instruct-v0.1",
                    huggingfacehub_api_token=hf_api_key,
                    temperature=0.7
                )
            except Exception as e:
                print(f"Mixtral initialization failed: {e}")
                return None

    def query(self, input_text):
        """
        process query with RAG or fallback to static response using cosine similarity.
        """
        if self.rag_chain:
            try:
                result = self.rag_chain.invoke(input_text)
                return result['result'].strip()
            except Exception as e:
                print(f"RAG query failed: {e}")

        # fallback: Static response using cosine similarity
        if self.embeddings and self.issue_embeddings is not None:
            try:
                query_embedding = np.array(self.embeddings.embed_query(input_text))
                similarities = cosine_similarity([query_embedding], self.issue_embeddings)[0]
                best_match_idx = np.argmax(similarities)
                if similarities[best_match_idx] > 0.5:  # Threshold for relevance
                    return self.df.iloc[best_match_idx]['Tech_Response']
            except Exception as e:
                print(f"Static fallback embedding failed: {e}")

        #last resort: Keyword match or default
        try:
            for _, row in self.df.iterrows():
                if input_text.lower() in row['Customer_Issue'].lower():
                    return row['Tech_Response']
            return self.df['Tech_Response'].iloc[0] if not self.df.empty else "Sorry, I couldn't find a matching response. Please try again or contact support."
        except Exception as e:
            return f"Static fallback failed: {str(e)}. Please contact support."