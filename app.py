import os
import httpx
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Configuration (Replace with your actual keys or use env variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key")
MCP_SERVER_BASE_URLS = [
    'https://api.merakle.ai/v1/b/mcp/messages'
]

# 1. Define Structured Output Schema
class WhatsAppResponse(BaseModel):
    responseText: str = Field(..., description="AI's message to the user; can be empty if using a WhatsApp template")
    responseWATemplate: str = Field(..., description="WhatsApp template ID to respond with, if applicable")
    saveDataVariable: str = Field(..., description="Variable name (e.g., 'contact_status') to save user's response data, if needed")
    saveDataValue: str = Field(..., description="Value of data to be saved for saveDataVariable, if applicable")
    waTemplateParams: List[str] = Field(..., description="An array of parameters to fill placeholders in the WhatsApp template, if applicable")

# 2. Define MCP Tool Logic (Mimicking callMCPServer from JS)
async def call_mcp_server(name: str, arguments: Dict[str, Any]) -> Any:
    """
    Helper function to call the MCP server for tool execution via JSON-RPC.
    """
    mcp_server_url = MCP_SERVER_BASE_URLS[0]
    print(f"callMCPServer: Executing tool {name} to {mcp_server_url}")
    
    payload = {
        "jsonrpc": "2.0",
        "id": "1",  # Static ID for simplicity or generate UUID
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments
        }
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(mcp_server_url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                print(f"MCP error: {data['error']}")
                return {"error": data["error"].get("message", "Unknown MCP error")}
            
            return data.get("result", {}).get("content", [])
        except Exception as e:
            print(f"MCP request failed: {e}")
            return {"error": str(e)}

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
    # Ensure IDs are strings for consistency
    task_id = str(request.taskId)
    account_id = str(request.accountId)
    campaign_id = str(request.campaignId)
    
    print(f"\n--- New Request: Task ID {task_id} ---")
    print(f"Account: {account_id} | Campaign: {campaign_id}")
    
    try:
        model_id = request.templateSettings.get("model", "gpt-4o-mini")
        temperature = request.templateSettings.get("temperature", 0.7)
        base_system_prompt = request.templateSettings.get("callprompt", "You are a helpful assistant.")
        enable_tools = request.templateSettings.get("campaign_settings", {}).get("enable_merakle_knowledge", False)
        
        print(f"Model: {model_id} | Temperature: {temperature}")
        print(f"Tools Enabled: {enable_tools}")

        tools = []
        if enable_tools:
            tools.append(call_mcp_server)

        # 4. Initialize Agno Agent
        from agno.utils.log import logger
        
        agent = Agent(
            model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, temperature=temperature),
            instructions=[
                base_system_prompt,
                f"The merakle_call_id for this call is {task_id}.",
                "Respond naturally to the user."
            ],
            tools=tools,
            output_schema=WhatsAppResponse,
            markdown=False,
        )

        # Inject chat history into agent memory
        print(f"Injecting {len(request.chatHistory)} messages into memory...")
        for msg in request.chatHistory[:-1]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                agent.memory.add_user_message(content)
            else:
                agent.memory.add_assistant_message(content)

        # 5. Run Agent
        last_user_message = request.chatHistory[-1]["content"] if request.chatHistory else "Hello"
        print(f"Last User Message: {last_user_message}")
        
        print("Agent is thinking/calling tools...")
        response = await agent.arun(last_user_message)
        
        # Agno's structured output
        if isinstance(response.content, WhatsAppResponse):
            print("Successfully generated structured response.")
            print(f"Response Text: {response.content.responseText}")
            if response.content.responseWATemplate:
                print(f"WA Template: {response.content.responseWATemplate} | Params: {response.content.waTemplateParams}")
            return response.content
        else:
            print("Warning: Response content was not an instance of WhatsAppResponse. Falling back.")
            return WhatsAppResponse(
                responseText=str(response.content),
                responseWATemplate="",
                saveDataVariable="",
                saveDataValue="",
                waTemplateParams=[]
            )

    except Exception as e:
        print(f"ERROR in /wa-agent: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3007)
