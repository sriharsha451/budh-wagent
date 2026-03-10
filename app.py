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
MCP_SERVER_URL = "https://api.merakle.ai/v1/b/mcp/messages"

# Reusable HTTP client
http_client = httpx.AsyncClient(timeout=30)


# -------------------------------------------------------
# 1. Structured Output Schema
# -------------------------------------------------------

class WhatsAppResponse(BaseModel):
    responseText: str = Field(..., description="AI's message to the user")
    responseWATemplate: str = Field(..., description="WhatsApp template ID if used")
    saveDataVariable: str = Field(..., description="Variable name to save user data")
    saveDataValue: str = Field(..., description="Value for the saved variable")
    waTemplateParams: List[str] = Field(..., description="Parameters for template placeholders")


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
# 3. Python-defined Tools (calling MCP internally)
# -------------------------------------------------------

@tool(
    name="merakle_demo_get_service_request_id",
    description="Generates a unique 7-digit service request ID. Call this tool only once to get a new ID, and use the ID directly in your response to the user."
)
async def merakle_demo_get_service_request_id() -> str: # Changed return type hint to str

    logger.info("Executing tool: merakle_demo_get_service_request_id")

    # Call the MCP server
    mcp_result = await call_mcp_server(
        "tools/call",
        {
            "name": "merakle_demo_get_service_request_id",
            "arguments": {}
        }
    )

    # Extract the ID from the MCP result for clarity to the LLM
    # Assuming the MCP server returns the ID in a key like "id" or "output"
    if isinstance(mcp_result, dict):
        service_request_id = mcp_result.get("id") or mcp_result.get("output")
        if service_request_id:
            logger.info(f"Generated Service Request ID: {service_request_id}")
            return str(service_request_id) # Ensure it's a string
        else:
            logger.error(f"MCP result for get_service_request_id did not contain an ID: {mcp_result}")
            return "Error: Could not generate a service request ID."
    else:
        logger.error(f"Unexpected MCP result type for get_service_request_id: {type(mcp_result)}")
        return "Error: Could not generate a service request ID."


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

        # Extract settings
        ts = request.templateSettings
        model_id = ts.get("model", "gpt-4o-mini")
        temperature = ts.get("temperature", 0.7)
        base_prompt = ts.get("callprompt", "You are a helpful assistant.")

        enable_tools = ts.get("campaign_settings", {}).get(
            "enable_merakle_knowledge", False
        )

        # -------------------------------------------------------
        # 5. Register Tools (Python-defined)
        # -------------------------------------------------------

        agno_tools = []

        if enable_tools:
            logger.info("Registering MCP-backed tools")

            agno_tools = [
                merakle_demo_get_service_request_id
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
        )

        # -------------------------------------------------------
        # 7. Format Chat History
        # -------------------------------------------------------

        chat_history_text = ""
        start_index = 0

        if request.chatHistory and request.chatHistory[0].get("role") == "system":
            chat_history_text += f"System Instructions: {request.chatHistory[0]['content']}\n\n"
            start_index = 1
        
        chat_history_text += "Conversation History:\n"

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
            waTemplateParams=[]
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
