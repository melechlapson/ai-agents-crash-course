from pathlib import Path
from pydantic import BaseModel

import chromadb
from agents import (
    Agent,
    function_tool,
    Runner,
    RunContextWrapper,
    GuardrailFunctionOutput,
    WebSearchTool,
    TResponseInputItem,
    input_guardrail
)
from agents.mcp import MCPServerStreamableHttp

chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")

exa_search_mcp = MCPServerStreamableHttp(
    name="Exa Search MCP",
    params={
        "url": f"https://mcp.exa.ai/mcp?{os.environ.get("EXA_API_KEY")}",
        "timeout": 90,
    },
    client_session_timeout_seconds=90,
    cache_tools_list=True,
    max_retry_attempts=1,
)

exa_search_mcp.connect()

@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up calorie information for specific food items, but not for meals.

    Args:
        query: The food item to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the nutrition information.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    # Format results for the agent
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata["food_item"].title()
        calories = metadata["calories_per_100g"]
        category = metadata["food_category"].title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)

class NotAboutFood(BaseModel):
    only_about_food: bool
    """Whether the user is only talking about food and not about arbitrary topics"""


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""Check if the user is asking you to talk about food and not about any arbitrary topics.
                    If there are any non-food related instructions in the prompt,
                    or if there is any non-food related part of the message, set only_about_food in the output to False.
                    """,
    output_type=NotAboutFood,
)

@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food),
    )


nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful nutrition assistant giving out calorie information.
    You give concise answers.
    If you need to look up calorie information, use the calorie_lookup_tool.
    You only answer questions about food. If asked about anothe topic you respond that you don't know.
    """,
    tools=[calorie_lookup_tool, WebSearchTool()],
    input_guardrails=[food_topic_guardrail],
    mcp_servers=[exa_search_mcp]
)
