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
from datetime import datetime
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
    responseText: Optional[str] = Field(None, description="AI's message/response to the user if applicable")
    responseWATemplate: Optional[str] = Field(None, description="WhatsApp template ID to respond with, if applicable")
    saveDataVariable: Optional[str] = Field(None, description="Variable name (e.g., 'contact_status') to save user's response data, if needed")
    saveDataValue: Optional[str] = Field(None, description="Value of data to be saved for saveDataVariable, if applicable")
    waTemplateParams: List[str] = Field(default_factory=list, description="An array of parameters to fill placeholders in the WhatsApp template, if applicable")
    waTemplateContent: Optional[str] = Field(None, description="Whatsapp template content, if applicable")
    fileAssetId: Optional[str] = Field(None, description="Asset ID of file to send the user, if applicable")
    isEndOfConversation: bool = Field(
        False, 
        description="Set to true if there are no more questions to ask the user and the conversation has reached its conclusion."
    )


class ToolParameter(BaseModel):
    name: str
    type: str


class ToolInfo(BaseModel):
    tool_name: str
    description: Optional[str]
    params: List[ToolParameter]


# -------------------------------------------------------
# 1.1. Validator Instructions
# -------------------------------------------------------

VALIDATOR_INSTRUCTIONS = """
OUTPUT JSON CONSTRAINTS:
  1) In output json, responseText and responseWATemplate are mutually exclusive props. 
     Only 1 can be set at a time. 
     If responseWATemplate is set then waTemplateParams and waTemplateContent also needs to be set.
  2) Always return a structured JSON object in the following exact format:

        {
        "responseText": "string | null",
        "responseWATemplate": "string | null",
        "saveDataVariable": "string | null",
        "saveDataValue": "string | null",
        "waTemplateParams": ["string"],
        "waTemplateContent": "string | null",
        "fileAssetId": "string | null",
        "isEndOfConversation": true
        }


        ### Field Rules:

        * "responseText" -> Use when sending a plain text reply
        * "responseWATemplate" -> Use when sending a WhatsApp template ID
        * "saveDataVariable" and "saveDataValue" -> Use together when storing user data
        * "waTemplateParams" -> Always return an array (use [] if not applicable)
        * "waTemplateContent" -> Rendered message content for the template
        * "fileAssetId" -> Use ONLY when sending a file to the user
        * "isEndOfConversation" -> true only when the conversation is fully complete, otherwise false

        ---

        ### 🚫 Mutual Exclusivity Rule (VERY IMPORTANT):

        * "responseText" and "responseWATemplate" are **mutually exclusive**
        * If "responseText" is NOT null -> "responseWATemplate" MUST be null
        * If "responseWATemplate" is NOT null -> "responseText" MUST be null
        * NEVER set both fields at the same time
        * NEVER leave both as null unless only sending file or save_data without user message

        ---

        ### 📁 File Handling Rule (VERY IMPORTANT):
        "fileAssetId" must be set ONLY when the AI is sending a file to the user
        If the user sends a file:
        DO NOT copy or reuse that file ID
        DO NOT set "fileAssetId" unless explicitly responding with a file
        In all other cases, set "fileAssetId" to null          

        --- 

        ### Strict Rules:

        * Always include ALL fields in every response
        * Do NOT remove or rename any fields
        * Use null for fields that are not applicable
        * "waTemplateParams" must always be an array (never null)
        * Do NOT include any extra fields
        * Do NOT include any text outside the JSON
        * Ensure valid JSON (double quotes only, no trailing commas)

        ---

  3) If the Chat Agent Workflow Summary specifies saving a variable:

    * You MUST populate "saveDataVariable" and "saveDataValue" exactly as instructed
    * Do NOT modify variable names

    ---

    15) Each response must be independent:

    * Do NOT carry forward values from previous turns
    * Always explicitly reset unused fields to null (or [] for arrays)

    ---

    ### Examples:

    ✅ Valid (Text response):
    {
    "responseText": "Hi! I am Mac. What's your name?",
    "responseWATemplate": null,
    "saveDataVariable": null,
    "saveDataValue": null,
    "waTemplateParams": [],
    "waTemplateContent": null,
    "fileAssetId": null,
    "isEndOfConversation": false
    }


    ✅ Valid (Template response):
    {
    "responseText": null,
    "responseWATemplate": "abcjshfs-38kj4f-fslkfhs8-fsifh9",
    "saveDataVariable": "contact_name",
    "saveDataValue": "Alex",
    "waTemplateParams": ["Alex"],
    "waTemplateContent": "Thanks Alex, would you like to know about Textgen?",
    "fileAssetId": null,
    "isEndOfConversation": false
    }


    ❌ Invalid (DO NOT DO THIS):
    {
    "responseText": "Hello",
    "responseWATemplate": "template_123"
    }
"""


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
        data = response.json()
        print("\n--- KNOWLEDGE BASE RESPONSE ---")
        print(data)
        print("-------------------------------\n")
        return data
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


def get_tools(campaign_id: str, tool_cache: dict) -> List[Any]:
    """Defines and returns the list of tools available for the agent."""
    
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
        json_res = await search_knowledge_base(campaign_id, query)
        import json
        return json.dumps(json_res, indent=2)

    return [
        merakle_demo_get_service_request_id,
        search_knowledge
    ]


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


@app.get("/discovery/tools", response_model=List[ToolInfo])
async def discover_tools_endpoint():
    """Returns a list of all tools available for the agent."""
    try:
        # Get tools using dummy values for metadata extraction
        agno_tools = get_tools(campaign_id="discovery", tool_cache={})
        
        tools_list = []
        for t in agno_tools:
            params = []
            # DEBUG: Print tool attributes to find the original function
            print(f"DEBUG: Tool {t.name} type: {type(t)}")
            print(f"DEBUG: Tool {t.name} dir: {dir(t)}")
            
            # Inspect parameters if available
            import inspect
            # Try various attributes Agno might use to store the original function
            func = getattr(t, "entry", None) or getattr(t, "function", None) or getattr(t, "original_function", None)
            
            if not func and callable(t):
                func = t

            if func:
                sig = inspect.signature(func)
                for name, param in sig.parameters.items():
                    # Skip 'self' or other common internal params
                    if name in ("self", "cls"):
                        continue
                    params.append(ToolParameter(
                        name=name,
                        type=str(param.annotation.__name__) if hasattr(param.annotation, "__name__") else str(param.annotation)
                    ))
            
            tools_list.append(ToolInfo(
                tool_name=t.name,
                description=t.description,
                params=params
            ))
        
        return tools_list

    except Exception as e:
        logger.exception(f"Error in /discovery/tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wa-agent", response_model=WhatsAppResponse)
async def run_agent_endpoint(request: AgentRequest):

    try:
        task_id = str(request.taskId)
        logger.info(f"--- New Request: Task {task_id} ---")

        # Request-scoped tool cache
        tool_cache = {}

        # Extract settings
        ts = request.templateSettings
        model_id = ts.get("model", "gpt-4.1-mini")
        temperature = ts.get("temperature", 0)
        base_prompt = ts.get("callprompt", "You are a helpful assistant.")

        enable_tools = ts.get("campaign_settings", {}).get(
            "enable_merakle_knowledge", False
        )

        # -------------------------------------------------------
        # 5. Register Tools
        # -------------------------------------------------------

        agno_tools = []
        if enable_tools:
            logger.info("Registering MCP-backed tools")
            agno_tools = get_tools(str(request.campaignId), tool_cache)

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
                f"The current date and time is {datetime.now()}.",
                f"The merakle_call_id for this call is {task_id}.",
                "Respond naturally to the user.",
                "Use tools when necessary.",
                "If a tool fails, inform the user and move on.",
                "Do not retry a tool more than once."
            ],
            tools=agno_tools,
            output_schema=WhatsAppResponse,
            markdown=False,
            tool_call_limit=5, # Increased limit to allow processing tool results.

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
        # 8. Run Agent with Critique Loop
        # -------------------------------------------------------

        response = await agent.arun(full_prompt)
        
        # Max retries for self-correction
        max_retries = 2
        current_attempt = 0
        
        final_validated_output = None

        while current_attempt < max_retries:
            logger.info(f"Running Validator Agent (Attempt {current_attempt + 1})...")
            
            validator_agent = Agent(
                model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY, temperature=0),
                instructions=[
                    "You are a strict output validator.",
                    "Your job is to check the Main Agent's output against the JSON CONSTRAINTS.",
                    "If the output is valid, return it exactly.",
                    "If it is invalid (e.g., both responseText and responseWATemplate are set), "
                    "use the Conversation History and Tool Results to determine the correct intent and fix the JSON.",
                    VALIDATOR_INSTRUCTIONS
                ],
                output_schema=WhatsAppResponse,
            )

            main_output_str = ""
            if isinstance(response.content, WhatsAppResponse):
                main_output_str = response.content.model_dump_json()
            else:
                main_output_str = str(response.content)

            # Capture tool results from the agent's run
            tool_results_str = ""
            if hasattr(response, "messages"):
                for m in response.messages:
                    if hasattr(m, "role") and m.role == "tool":
                        tool_results_str += f"Tool Result: {m.content}\n"

            # Pass history, tool results, and the output to the validator
            validation_payload = (
                f"CONVERSATION HISTORY:\n{full_prompt}\n\n"
                f"TOOL RESULTS:\n{tool_results_str if tool_results_str else 'No tools were called.'}\n\n"
                f"MAIN AGENT OUTPUT TO VALIDATE:\n{main_output_str}"
            )
            
            validated_response = await validator_agent.arun(validation_payload)
            
            # Check if the validator successfully produced a valid object
            if isinstance(validated_response.content, WhatsAppResponse):
                final_validated_output = validated_response.content
                logger.info("Validation successful.")
                break
            else:
                current_attempt += 1
                logger.warning(f"Validation attempt {current_attempt} failed. Retrying main agent...")
                retry_prompt = full_prompt + f"\n\nCRITIQUE: Your previous response was invalid. Ensure you follow the JSON constraints strictly. Error: {str(validated_response.content)}"
                response = await agent.arun(retry_prompt)

        # -------------------------------------------------------
        # 9. Return Structured Response
        # -------------------------------------------------------

        if final_validated_output:
            return final_validated_output

        # Final fallback if loop fails
        return WhatsAppResponse(
            responseText=str(response.content),
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
