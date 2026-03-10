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
MCP_SERVER_URL = 'https://api.merakle.ai/v1/b/mcp/messages'

# 1. Structured Output Schema
class WhatsAppResponse(BaseModel):
    responseText: str = Field(..., description="AI's message to the user")
    responseWATemplate: str = Field(..., description="WhatsApp template ID if used")
    saveDataVariable: str = Field(..., description="Variable name to save user data")
    saveDataValue: str = Field(..., description="Value for the saved variable")
    waTemplateParams: List[str] = Field(..., description="Parameters for template placeholders")

# 2. MCP Bridge Logic
async def call_mcp_server(method: str, params: dict) -> Any:
    """Generic JSON-RPC caller for the MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(MCP_SERVER_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                logger.error(f"MCP error in {method}: {data['error']}")
                return {"error": data["error"].get("message", "Unknown MCP error")}
            return data.get("result", {})
        except Exception as e:
            logger.exception(f"MCP {method} failed: {e}")
            return {"error": str(e)}

async def fetch_mcp_tools() -> List[Dict]:
    """Fetches the list of available tools from the MCP server."""
    result = await call_mcp_server("tools/list", {})
    return result.get("tools", [])

def build_agno_tool(mcp_tool_def: Dict):
    """Dynamically builds an Agno Tool from an MCP tool definition using the @tool decorator."""
    tool_name = mcp_tool_def["name"]
    tool_desc = mcp_tool_def.get("description", "No description provided.")

    @tool(name=tool_name, description=tool_desc)
    async def mcp_executor(arguments: Dict[str, Any] = {}) -> Dict[str, Any]:
        # This is the function the agent actually calls
        logger.info(f"Executing MCP Tool: {tool_name} with args: {arguments}")
        result = await call_mcp_server("tools/call", {"name": tool_name, "arguments": arguments})
        return result

    return mcp_executor

# 3. FastAPI App
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

        # Extract settings
        ts = request.templateSettings
        model_id = ts.get("model", "gpt-4o-mini")
        temperature = ts.get("temperature", 0.7)
        base_prompt = ts.get("callprompt", "You are a helpful assistant.")
        enable_tools = ts.get("campaign_settings", {}).get("enable_merakle_knowledge", False)

        # 4. Build Tools List
        agno_tools = []
        if enable_tools:
            logger.info("Fetching MCP tools...")
            mcp_tools_defs = await fetch_mcp_tools()
            for t_def in mcp_tools_defs:
                agno_tools.append(build_agno_tool(t_def))
            logger.info(f"Registered {len(agno_tools)} tools from MCP server.")

        # 5. Initialize Agent
        agent = Agent(
            model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, temperature=temperature),
            instructions=[
                base_prompt,
                f"The merakle_call_id for this call is {task_id}.",
                "Respond naturally to the user.",
                "If a tool fails, inform the user and move on. Do not retry more than once."
            ],
            tools=agno_tools,
            output_schema=WhatsAppResponse,
            markdown=False,
            num_calls=10 # Safety limit
        )

        # 6. Format Chat History
        chat_history_text = ""
        start_index = 0
        if request.chatHistory and request.chatHistory[0].get("role") == "system":
            chat_history_text += f"System Instructions: {request.chatHistory[0]['content']}\n\n"
            start_index = 1

        for msg in request.chatHistory[start_index:-1]:
            role = msg.get("role", "user").title()
            chat_history_text += f"{role}: {msg.get('content')}\n"

        last_msg = request.chatHistory[-1]["content"] if request.chatHistory else "Hi"
        full_prompt = chat_history_text + f"User: {last_msg}"

        # Debug logs
        print(f"\n--- FINAL PROMPT ---\n{full_prompt}\n")

        # 7. Run Agent
        response = await agent.arun(full_prompt)

        # 8. Return Response
        if isinstance(response.content, WhatsAppResponse):
            return response.content
        
        # Fallback
        return WhatsAppResponse(
            responseText=str(response.content),
            responseWATemplate="",
            saveDataVariable="",
            saveDataValue="",
            waTemplateParams=[]
        )

    except Exception as e:
        logger.exception(f"Error in /wa-agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3007)
