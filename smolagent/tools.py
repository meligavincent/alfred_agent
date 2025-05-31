from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from retrievers import docs
class GuestInfoRetrieverTool(Tool):
    name = "GuestInfoRetrieverToll"
    description = "Retrieves guest information from a database"
    inputs = {
        "query":{
            "type": "string",
            "description": "The name or relation of the guesst you want information about"
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query:str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found"
        
guest_info_tool = GuestInfoRetrieverTool(docs)