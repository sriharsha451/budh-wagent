import os
import httpx
import uuid
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_API_ENDPOINT = os.getenv("KNOWLEDGE_API_ENDPOINT")
KNOWLEDGE_API_KEY = os.getenv("KNOWLEDGE_API_KEY")
MCP_SERVER_URL = "https://api.merakle.ai/v1/b/mcp/messages"

# Reusable HTTP client
http_client = httpx.AsyncClient(timeout=30)


# -------------------------------------------------------
# 1. Structured Output Schema
# -------------------------------------------------------

class WhatsAppResponse(BaseModel):
    responseText: str = Field(..., description="AI's message/response to the user if applicable")
    responseWATemplate: str = Field(..., description="WhatsApp template ID to respond with, if applicable")
    saveDataVariable: str = Field(..., description="Variable name (e.g., 'contact_status') to save user's response data, if needed")
    saveDataValue: str = Field(..., description="Value of data to be saved for saveDataVariable, if applicable")
    waTemplateParams: List[str] = Field(..., description="An array of parameters to fill placeholders in the WhatsApp template, if applicable")

    isEndOfConversation: bool = Field(
        ..., 
        description="Set to true if there are no more questions to ask the user and the conversation has reached its conclusion."
    )


# -------------------------------------------------------
# 2. MCP Bridge Logic
# -------------------------------------------------------

async def call_mcp_server(method: str, params: dict) -> Any:
    """Generic JSON-RPC caller for the MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params
    }

    try:
        response = await http_client.post(MCP_SERVER_URL, json=payload)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            logger.error(f"MCP error in {method}: {data['error']}")
            return {"error": data["error"].get("message", "Unknown MCP error")}

        return data.get("result", {})

    except Exception as e:
        logger.exception(f"MCP {method} failed: {e}")
        return {"error": str(e)}


# -------------------------------------------------------
# 3. Generic MCP Tool Executor (with Caching)
# -------------------------------------------------------

async def search_knowledge_base(campaign_id: str, query: str) -> Any:
    """Search knowledgebase for answers to user's queries."""
    url = f"{KNOWLEDGE_API_ENDPOINT}/text-campaigns/{campaign_id}/knowledge/search"
    headers = {
        "x-api-key": f"{KNOWLEDGE_API_KEY}"
    }
    payload = {
        "query": query
    }

    try:
        response = await http_client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.exception(f"Knowledge base search failed: {e}")
        return {"error": str(e)}


async def execute_mcp_tool(name: str, arguments: dict, cache: dict) -> str:
    """
    Executes an MCP tool with request-scoped caching.
    Standardizes response handling to be generic across different tools.
    """
    # Create a unique cache key based on tool name and arguments
    import json
    cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
    
    if cache_key in cache:
        logger.info(f"Cache HIT for tool: {name}")
        return cache[cache_key]

    logger.info(f"Cache MISS for tool: {name}. Executing...")

    # Call the MCP server
    mcp_result = await call_mcp_server(
        "tools/call",
        {
            "name": name,
            "arguments": arguments
        }
    )

    # Standardized response extraction
    # 1. If MCP returns an error
    if isinstance(mcp_result, dict) and "error" in mcp_result:
        return f"Error from tool {name}: {mcp_result['error']}"

    # 2. Try common MCP/Merakle result keys
    result_data = None
    if isinstance(mcp_result, dict):
        # Check for 'content' (standard MCP), then 'id', 'output', 'result', etc.
        result_data = (
            mcp_result.get("content") or 
            mcp_result.get("id") or 
            mcp_result.get("output") or 
            mcp_result.get("result") or 
            mcp_result
        )
    else:
        result_data = mcp_result

    # 3. Convert to a clean string for the LLM
    final_result = str(result_data)
    
    # Store in cache and return
    cache[cache_key] = final_result
    return final_result


# -------------------------------------------------------
# 4. FastAPI App
# -------------------------------------------------------

app = FastAPI()


class AgentRequest(BaseModel):
    accountId: Any
    campaignId: Any
    taskId: Any
    chatHistory: List[Dict[str, str]]
    templateSettings: Dict[str, Any]


@app.post("/wa-agent", response_model=WhatsAppResponse)
async def run_agent_endpoint(request: AgentRequest):

    try:
        task_id = str(request.taskId)
        logger.info(f"--- New Request: Task {task_id} ---")

        # Request-scoped tool cache
        tool_cache = {}

        # Extract settings
        ts = request.templateSettings
        model_id = ts.get("model", "gpt-4o-mini")
        temperature = ts.get("temperature", 0)
        base_prompt = ts.get("callprompt", "You are a helpful assistant.")

        enable_tools = ts.get("campaign_settings", {}).get(
            "enable_merakle_knowledge", False
        )

        # -------------------------------------------------------
        # 5. Register Tools (Generic wrappers)
        # -------------------------------------------------------

        agno_tools = []

        if enable_tools:
            logger.info("Registering MCP-backed tools")

            @tool(
                name="merakle_demo_get_service_request_id",
                description="Generates a unique 7-digit service request ID. Call this tool only once to get a new ID, and use the ID directly in your response to the user."
            )
            async def merakle_demo_get_service_request_id() -> str:
                """Generic wrapper for service request ID generation."""
                return await execute_mcp_tool(
                    "merakle_demo_get_service_request_id", 
                    {}, 
                    tool_cache
                )

            @tool(
                name="search_knowledge",
                description="Search knowledgebase for answers to user's queries"
            )
            async def search_knowledge(query: str) -> str:
                """Search knowledgebase for answers to user's queries."""
                campaign_id = str(request.campaignId)
                json_res = await search_knowledge_base(campaign_id, query)
                import json
                return json.dumps(json_res, indent=2)

            agno_tools = [
                merakle_demo_get_service_request_id,
                search_knowledge
            ]

        # -------------------------------------------------------
        # 6. Initialize Agent
        # -------------------------------------------------------

        agent = Agent(
            model=OpenAIChat(
                id=model_id,
                api_key=OPENAI_API_KEY,
                temperature=temperature
            ),
            instructions=[
                base_prompt,
                f"The merakle_call_id for this call is {task_id}.",
                "Respond naturally to the user.",
                "Use tools when necessary.",
                "If a tool fails, inform the user and move on.",
                "Do not retry a tool more than once."
            ],
            tools=agno_tools,
            output_schema=WhatsAppResponse,
            markdown=False,
            tool_call_limit=1, # The Agent will not perform more than one tool call.

        )

        # -------------------------------------------------------
        # 7. Format Chat History
        # -------------------------------------------------------

        chat_history_text = ""
        start_index = 0

        if request.chatHistory and request.chatHistory[0].get("role") == "system":
            chat_history_text += f"System Instructions: {request.chatHistory[0]['content']}\n\n"
            start_index = 1
        
        chat_history_text += "\n\nConversation History:\n"

        for msg in request.chatHistory[start_index:-1]:
            role = msg.get("role", "user").title()
            chat_history_text += f"{role}: {msg.get('content')}\n"

        last_msg = request.chatHistory[-1]["content"] if request.chatHistory else "Hi"

        full_prompt = chat_history_text + f"User: {last_msg}"

        print("\n--- FINAL PROMPT ---\n")
        print(full_prompt)
        print("\n--------------------\n")

        # -------------------------------------------------------
        # 8. Run Agent
        # -------------------------------------------------------

        response = await agent.arun(full_prompt)

        # -------------------------------------------------------
        # 9. Return Structured Response
        # -------------------------------------------------------

        if isinstance(response.content, WhatsAppResponse):
            return response.content

        return WhatsAppResponse(
            responseText=str(response.content),
            responseWATemplate="",
            saveDataVariable="",
            saveDataValue="",
            waTemplateParams=[],
            isEndOfConversation=False
        )

    except Exception as e:
        logger.exception(f"Error in /wa-agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------
# 10. Run Server
# -------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3007)
