import os
import httpx
import uuid
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from agno.agent import Agent, Tool
from agno.models.openai import OpenAIChat
from agno.utils.log import logger

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
MCP_SERVER_BASE_URLS = ['https://api.merakle.ai/v1/b/mcp/messages']

# 1. Structured Output Schema
class WhatsAppResponse(BaseModel):
    responseText: str = Field(..., description="AI's message to the user")
    responseWATemplate: str = Field(..., description="WhatsApp template ID if used")
    saveDataVariable: str = Field(..., description="Variable name to save user data")
    saveDataValue: str = Field(..., description="Value for the saved variable")
    waTemplateParams: List[str] = Field(..., description="Parameters for template placeholders")

# 2. MCP Tool Logic
async def call_mcp_server(name: str, arguments: Dict[str, Any]) -> Any:
    url = MCP_SERVER_BASE_URLS[0]
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),  # unique per request
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    }
    try:
        print(f"callMCPServer: Executing tool {name}")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                logger.error(f"MCP error: {data['error']}")
                return {"error": data["error"].get("message", "Unknown MCP error")}
            
            # Extract content from result
            result = data.get("result", {})
            return result.get("content", []) or result.get("output", [])
    except Exception as e:
        logger.exception(f"MCP request failed: {e}")
        return {"error": str(e)}

# Wrap MCP call as a Tool
mcp_tool = Tool(name="callMCPServer", func=call_mcp_server)

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
        account_id = str(request.accountId)
        campaign_id = str(request.campaignId)
        logger.info(f"New Request: Task {task_id} | Account {account_id} | Campaign {campaign_id}")

        template_settings = request.templateSettings
        model_id = template_settings.get("model", "gpt-4o-mini")
        temperature = template_settings.get("temperature", 0.7)
        base_system_prompt = template_settings.get("callprompt", "You are a helpful assistant.")
        enable_tools = template_settings.get("campaign_settings", {}).get("enable_merakle_knowledge", False)

        logger.info(f"Model: {model_id} | Temperature: {temperature} | Tools Enabled: {enable_tools}")

        # Tools list
        tools = [mcp_tool] if enable_tools else []

        # Initialize Agno Agent
        agent = Agent(
            model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, temperature=temperature),
            instructions=[
                base_system_prompt,
                f"The merakle_call_id for this call is {task_id}.",
                "Respond naturally to the user."
            ],
            tools=tools,
            output_schema=WhatsAppResponse,
            markdown=False
            # Removed show_tool_calls and add_history_to_messages due to environment compatibility
        )

        # Inject chat history into agent memory
        logger.info(f"Injecting {len(request.chatHistory)} messages into memory...")
        for msg in request.chatHistory[:-1]:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role == "user":
                agent.memory.add_user_message(content)
            else:
                agent.memory.add_assistant_message(content)

        # Last user message
        last_message = request.chatHistory[-1]["content"] if request.chatHistory else "Hello"

        logger.info(f"Running agent on last user message: {last_message}")
        response = await agent.arun(last_message)

        # Return structured response
        if isinstance(response.content, WhatsAppResponse):
            logger.info("Successfully generated structured response.")
            return response.content
        else:
            # Fallback for unexpected content
            logger.warning("Response content was not WhatsAppResponse instance. Falling back.")
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
