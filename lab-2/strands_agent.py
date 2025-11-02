from mcp import stdio_client, StdioServerParameters
from strands import Agent, tool
from strands.tools.mcp import MCPClient
from strands.models import BedrockModel
import boto3 
import os
import json
import requests
import re

@tool
def weather(lat, lon: float) -> str:
    """Get weather information for a given lat and lon

    Args:
        lat: latitude of the location
        lon: logitude of the location
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "current_weather": True
    }
    response = requests.get(url, params=params)
    return response.json()["current_weather"]

class StrandsAgent:

    def __init__(self):
        # Launch AWS Location Service MCP Server and create a client object
        env = {"FASTMCP_LOG_LEVEL": "ERROR"}
        env['AWS_SECRET_ACCESS_KEY']=os.getenv("AWS_SECRET_ACCESS_KEY")
        env['AWS_ACCESS_KEY_ID']=os.getenv("AWS_ACCESS_KEY_ID")
        env['AWS_REGION']=os.getenv("AWS_REGION")

        self.aws_location_srv_client = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command="uvx", 
                args=["awslabs.aws-location-mcp-server@latest"],
                env=env)
            ))
        self._server_context = self.aws_location_srv_client.__enter__()
        self.aws_location_srv_tools = self.aws_location_srv_client.list_tools_sync()

        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name='us-east-1'
        )
        # Specify Bedrock LLM for the Agent
        bedrock_model = BedrockModel(
            model_id="amazon.nova-lite-v1:0",
            boto_session=session
        )
        # Create a Strands Agent
        tools = self.aws_location_srv_tools
        tools.append(weather)
        self.agent = Agent(
            tools=tools, 
            model=bedrock_model,
        )


    '''
    Send the input to the agent, allowing it to handle tool selection and invocation. 
    The response will be generated after the selected LLM performs reasoning. 
    This approach is suitable when you want to delegate tool selection logic to the agent, and have a generic toolUse definition in Sonic ToolUse.
    Note that the reasoning process may introduce latency, so it's recommended to use a lightweight model such as Nova Lite.
    Sample parameters: input="largest zoo in Seattle?"
    '''
    def query(self, input):
        output = str(self.agent(input))
        if "<response>" in output and "</response>" in output:
            output = re.search(r"<response>(.*?)</response>", output, re.DOTALL)
        elif "<answer>" in output and "</answer>" in output:
            output = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        return output

    '''
    Invoke the tool directly and return the raw response without any reasoning.
    This approach is suitable when tool selection is managed within Sonic and the exact toolName is already known. 
    It offers lower query latency, as no additional reasoning is performed by the agent.
    Sample parameters: tool_name="search_places", input="largest zoo in Seattle"
    '''
    def call_tool(self, tool_name, input):
        if isinstance(input, str):
            input = json.loads(input)
        if "query" in input:
            input = input.get("query")

        tool_func = getattr(self.agent.tool, tool_name)
        return tool_func(query=input)

    def close(self):
        # Cleanup the MCP server context
        self.aws_location_srv_client.__exit__(None, None, None)