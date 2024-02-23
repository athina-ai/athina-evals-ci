"""
This is a sample llm application using RAG. 
We will use llama_index to generate a response for a given query. 
"""

import os
import openai
from typing import List
from dotenv import load_dotenv
from llama_index import download_loader
from llama_index import VectorStoreIndex, ServiceContext

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class Application(): 
    
    def __init__(self, openai_api_key):
        openai.api_key = OPENAI_API_KEY
        self.query_engine = self.create_query_engine()

    def create_query_engine(self):
        # create a llamaindex query engine
        WikipediaReader = download_loader("WikipediaReader")
        loader = WikipediaReader()
        documents = loader.load_data(pages=['Y Combinator'])
        vector_index = VectorStoreIndex.from_documents(
            documents, service_context=ServiceContext.from_defaults(chunk_size=512)
        )
        return vector_index.as_query_engine()

    def generate_response(self, query: str, top_k: int = 1) -> List[str]:
        """
        Generates a response for the given query.
        """
        contexts = []
        query_engine_response = self.query_engine.query(query)
        response = query_engine_response.response
        for c in query_engine_response.source_nodes:
            text = c.node.get_text()
            contexts.append(text)

        return contexts, response
    
if __name__ == "__main__":
    # Initialize the application
    app = Application(openai_api_key=OPENAI_API_KEY)
    print(app.generate_response("How much equity does YC take?"))
