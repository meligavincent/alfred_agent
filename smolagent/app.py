from smolagents import CodeAgent, InferenceClientModel
from alfred_agent.smolagent.tools import  guest_info_tool, search_tool, weather_info_tool, hub_stats_tool
# Initialise the Hugging Face model
model = InferenceClientModel()

# Create Alfre,our gala agent, with guest info tool
alfred = CodeAgent(tools=[guest_info_tool,search_tool,weather_info_tool,hub_stats_tool],model=model)

# Example query Alfred might receive during gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("Alfred's Response")
print(response)