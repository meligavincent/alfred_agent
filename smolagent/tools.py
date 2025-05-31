import random
from huggingface_hub import list_models
from smolagents import Tool,DuckDuckGoSearchTool
from langchain_community.retrievers import BM25Retriever
from alfred_agent.smolagent.retrievers import docs
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
        

class WeatherInfoTool(Tool):
    name = "WeatherInfoTool"
    description = "Retrieves weather information for a given location"
    inputs = {
        "location": {
            "type": "string",
            "description": "The location for which you want the weather information"
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        weather_conditions = [
            {"condition": "Rainy", "temp_c": 15},
            {"condition": "Clear", "temp_c": 25},
            {"condition": "Windy", "temp_c": 20}
        ]
        # Randomly select a weather condition
        data = random.choice(weather_conditions)
        return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"


class HubStatsTool(Tool):
    name = "HubStatsTool"
    description = "Fetches the most dowmloaded models from Hugging Face Hub"
    inputs = {
        "author":{
            "type": "string",
            "description": "The author of the models you want to fetch"
        }
    }

    output_type = "string"

    def forward(self,author: str):
        
        try:
            models = list(list_models(author=author, sort="downloads", direction=-1,limit=1))

            if models:
                model = models[0]
                return f"Most downloaded model by {author}: {model.modelId} with {model.downloads} downloads"
            else:
                return f"No models found for author {author}"
        except Exception as e:
            return f"Error fetching models for author {author}: {str(e)}"
    

# Initialise the DuckDuckGo search tool and the guest info retriever tool
search_tool = DuckDuckGoSearchTool()
guest_info_tool = GuestInfoRetrieverTool(docs)
weather_info_tool = WeatherInfoTool()
hub_stats_tool = HubStatsTool()