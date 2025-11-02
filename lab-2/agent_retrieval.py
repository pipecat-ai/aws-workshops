# agent_retrieval.py
import argparse
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.aws_nova_sonic import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from retrieval import search_health_info, summarize_search_results

# Load environment variables
load_dotenv(override=True)

async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )

async def retrieve_health_info(params: FunctionCallParams):
    query = params.arguments.get("query", "")
    if not query:
        await params.result_callback({"result": "No query provided."})
        return
    
    # Search for health information
    search_results = search_health_info(query)
    
    # Summarize the results
    summary = summarize_search_results(search_results, query)
    
    # Return the results
    await params.result_callback({
        "result": summary,
        "query": query,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })

# Define the weather function schema
weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the users location.",
        },
    },
    required=["location", "format"],
)

# Create a function schema for health information retrieval
health_info_function = FunctionSchema(
    name="retrieve_health_info",
    description="Retrieve health information about a specific topic or question",
    properties={
        "query": {
            "type": "string",
            "description": "The health-related query or question",
        },
    },
    required=["query"],
)

# Create tools schema with both functions
tools = ToolsSchema(standard_tools=[weather_function, health_info_function])

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Specify initial system instruction with health assistant capabilities
    system_instruction = (
        "You are a helpful health assistant designed to provide general health information. "
        "You can answer health-related questions and provide information on symptoms, treatments, "
        "and preventive measures. For specific medical questions, you can search for information "
        "using the retrieve_health_info function. "
        "Keep your responses short, generally two or three sentences for chatty scenarios. "
        "Always attribute information to sources when using search results. "
        "Remember that you are providing general information only, not medical advice. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    # Create the AWS Nova Sonic LLM service
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region=os.getenv("AWS_REGION"),  # as of 2025-05-06, us-east-1 is the only supported region
        voice_id="tiffany",  # matthew, tiffany, amy
    )

    # Register functions for function calls
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("retrieve_health_info", retrieve_health_info)

    # Set up context and context management
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Hello, I'd like to ask some health questions.",
            },
        ],
        tools=tools,
    )
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Configure the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])
        # Trigger the first assistant response
        await llm.trigger_assistant_response()

    # Handle client disconnection events
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
