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

# Load environment variables
load_dotenv(override=True)

# Import our StrandsAgent
from strands_agent import StrandsAgent

# Create a global StrandsAgent instance
strands_agent = StrandsAgent()

# Function to handle queries using our StrandsAgent
async def handle_query(params: FunctionCallParams):
    query = params.arguments.get("query", "")
    if not query:
        await params.result_callback({"result": "No query provided."})
        return
        
    response = strands_agent.query(query)
    if response:
        result = response.group(1) if hasattr(response, "group") else str(response)
        await params.result_callback({
            "result": result,
            "query": query,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })
    else:
        await params.result_callback({
            "result": "I couldn't process that request.",
            "query": query,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        })

# Create a function schema for the Strands agent query
query_function = FunctionSchema(
    name="handle_query",
    description="Delegates queries to a Strands agent that can access location and weather information",
    properties={
        "query": {
            "type": "string",
            "description": "The query to delegate to the Strands agent"
        }
    },
    required=["query"],
)

# Create tools schema
tools = ToolsSchema(standard_tools=[query_function])

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot with Strands agent integration")
    
    # Initialize services
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region=os.getenv("AWS_REGION"),  # as of 2025-05-06, us-east-1 is the only supported region
        voice_id="tiffany",  # matthew, tiffany, amy
    )
    
    # Register our handle_query function
    llm.register_function("handle_query", handle_query)
    
    # Specify initial system instruction
    system_instruction = (
        "You are a helpful health assistant designed to provide general health information. "
        "When users ask about location-based information or weather conditions, use the handle_query "
        "function to delegate the request to a Strands agent that has access to location and weather tools. "
        "Keep your responses short, generally two or three sentences. "
        "Remember that you are providing general information only, not medical advice. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )
    
    # Set up context and context management
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Hello, I'm interested in health information.",
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
