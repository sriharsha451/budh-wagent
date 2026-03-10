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
    accountId: str
    campaignId: str
    taskId: str
    chatHistory: List[Dict[str, str]]
    templateSettings: Dict[str, Any]

@app.post("/wa-agent", response_model=WhatsAppResponse)
async def run_agent_endpoint(request: AgentRequest):
    try:
        model_id = request.templateSettings.get("model", "gpt-4o-mini")
        temperature = request.templateSettings.get("temperature", 0.7)
        base_system_prompt = request.templateSettings.get("callprompt", "You are a helpful assistant.")
        
        # Enable Merakle knowledge (MCP tools) based on campaign settings
        enable_tools = request.templateSettings.get("campaign_settings", {}).get("enable_merakle_knowledge", False)
        
        # In Agno, we can define tools as functions
        # This is a simplified version; in a real scenario, you'd discover tools from the MCP server
        tools = []
        if enable_tools:
            # Note: For Agno, we'd typically register available tools here.
            # Since the JS code seems to dynamically call ANY tool requested by LLM, 
            # we'll provide a generic tool or use Agno's native MCP support if possible.
            # For now, we'll implement it as a function the agent can call.
            tools.append(call_mcp_server)

        # 4. Initialize Agno Agent with Memory to handle chatHistory
        from agno.memory.agent import AgentMemory
        from agno.memory.db.sqlite import SqliteMemoryDb
        from agno.utils.log import logger

        # Convert input chatHistory to Agno message format if needed, 
        # but Agno Agents usually manage history in their own memory.
        # For a stateless API call, we can pass the history to the agent's memory.
        agent = Agent(
            model=OpenAIChat(id=model_id, api_key=OPENAI_API_KEY, temperature=temperature),
            instructions=[
                base_system_prompt,
                f"The merakle_call_id for this call is {request.taskId}.",
                "Respond naturally to the user."
            ],
            tools=tools,
            output_schema=WhatsAppResponse,
            markdown=False,
            show_tool_calls=True,
            # We don't need a persistent DB for a stateless API call, 
            # so we use in-memory history.
            add_history_to_messages=True, 
        )

        # In Agno, we can manually add messages to the agent's memory before running
        for msg in request.chatHistory[:-1]:  # Add all but the last message to history
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                agent.memory.add_user_message(content)
            else:
                agent.memory.add_assistant_message(content)

        # 5. Run Agent with the last user message
        last_user_message = request.chatHistory[-1]["content"] if request.chatHistory else "Hello"
        response = await agent.arun(last_user_message)
        
        # Agno's structured output is in response.content (as a WhatsAppResponse instance)
        if isinstance(response.content, WhatsAppResponse):
            return response.content
        else:
            # Fallback if parsing fails (though Agno handles it)
            return WhatsAppResponse(
                responseText=str(response.content),
                responseWATemplate="",
                saveDataVariable="",
                saveDataValue="",
                waTemplateParams=[]
            )

    except Exception as e:
        print(f"Error running agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3007)
