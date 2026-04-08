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
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_API_ENDPOINT = os.getenv("KNOWLEDGE_API_ENDPOINT")
KNOWLEDGE_API_KEY = os.getenv("KNOWLEDGE_API_KEY")
MCP_SERVER_URL = "https://api.merakle.ai/v1/b/mcp/messages"
DEFAULT_MODEL = "gpt-4.1-mini"

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
    setNextWaitUntil: Optional[str] = Field(None, description="The timestamp to wait until before the next action, in ISO 8601 UTC format (e.g., '2026-03-30T10:00:00Z').")
    nextNode: Optional[str] = Field(None, description="The ID of the next node to transition to, if applicable")
    quickReplyOptions: List[str] = Field(default_factory=list, description="An array of strings representing quick reply buttons for the user, if applicable")
    isYesOrNoQuestion: bool = Field(False, description="Set to true if the responseText is a question that expects a yes or no answer.")
    isEndOfConversation: bool = Field(
        False, 
        description="Set to true if there are no more questions to ask the user and the conversation has reached its conclusion."
    )


class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Set to true if the Main Agent's output follows all JSON CONSTRAINTS and logical rules (or if you can fix it with simple adjustments).")
    critique: Optional[str] = Field(None, description="If the output is invalid and cannot be easily fixed, provide a detailed explanation of the violations.")
    fixed_response: Optional[WhatsAppResponse] = Field(None, description="The final, corrected WhatsAppResponse object. ALWAYS populate this with the best possible version of the output.")


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
OUTPUT JSON RULES:

1) Response Type (Required):
   - Exactly ONE of the following must be set (non-empty):
     • responseText
     • responseWATemplate
   - If BOTH are null/empty → fileAssetId MUST be set
    
2) WhatsApp Template Rule:
   - If responseWATemplate is set (non-empty):
     • waTemplateParams MUST be provided (array)
     • waTemplateContent MUST be provided (non-empty)

3) JSON Format (Always return ALL fields):

{
  "responseText": "string | null",
  "responseWATemplate": "string | null",
  "saveDataVariable": "string | null",
  "saveDataValue": "string | null",
  "waTemplateParams": ["string"],
  "waTemplateContent": "string | null",
  "fileAssetId": "string | null",
  "setNextWaitUntil": "string | null",
  "nextNode": "string | null",
  "quickReplyOptions": ["string"],
  "isYesOrNoQuestion": false,
  "isEndOfConversation": true
}

4) Field Usage:
   - responseText → plain message
   - responseWATemplate → WhatsApp template ID
   - saveDataVariable + saveDataValue → must be used together
   - waTemplateParams → always array (use [] if none)
   - quickReplyOptions → always array (use [] if none)
   - fileAssetId → ONLY when sending a file
   - setNextWaitUntil → ISO 8601 UTC format
   - isYesOrNoQuestion → true only for yes/no questions

5) File Rules:
   - Only set fileAssetId when sending a file
   - Never reuse user-uploaded file IDs
   - Otherwise keep it null

6) Strict Constraints:
   - Always include ALL fields
   - No extra fields
   - No text outside JSON
   - Valid JSON only (double quotes, no trailing commas)

7) Data Saving:
   - If instructed, MUST set saveDataVariable and saveDataValue exactly

8) Stateless Responses:
   - Each response is independent
   - Do not reuse previous values
   - Reset unused fields to null or []

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


def get_tools(campaign_id: str, tool_cache: dict, chat_history: List[Dict[str, Any]] = None, default_model: str = DEFAULT_MODEL) -> List[Any]:
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
        description="Searches the knowledgebase for answers to the user's query. This tool should be invoked only when the user asks for information that requires searching the knowledgebase."
    )
    async def search_knowledge(query: str) -> str:
        """Searches the knowledgebase for answers to the user's query. This tool should be invoked only when the user asks for information that requires searching the knowledgebase."""
        json_res = await search_knowledge_base(campaign_id, query)
        import json
        return json.dumps(json_res, indent=2)

    @tool(
        name="textgen_trigger_node_wait",
        description="Returns a future timestamp based on the user's specified wait criteria. This tool should only be called when explicitly instructed by a Step in the workflow."
    )
    async def textgen_trigger_node_wait(merakle_call_id: str, query: str) -> str:
        """Returns a future timestamp based on the user's specified wait criteria. This tool should only be called when explicitly instructed by a Step in the workflow."""
        print(f"\n--- TOOL USER QUERY (textgen_trigger_node_wait) ---\n{query}\n--------------------------------------------------\n")
        
        ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        today_date = ist_now.strftime("%Y-%m-%d")
        tomorrow_date = (ist_now + timedelta(days=1)).strftime("%Y-%m-%d")
        current_date_time = ist_now.strftime("%Y-%m-%dT%H:%M:%S")

        # Format history for the prompt
        history_str = ""
        if chat_history:
            start_index = 0
            if chat_history[0].get("role") == "system":
                start_index = 1
            
            history_str += "\n\nConversation History:\n"
            for msg in chat_history[start_index:-1]:
                role = msg.get("role", "user").title()
                history_str += f"{role}: {msg.get('content')}\n"

            last_msg = chat_history[-1]["content"] if chat_history else "Hi"
            history_str += f"User: {last_msg}"

        prompt = f"""
                
        Current Date and Time : {current_date_time}
        Today's Date : {today_date}
        Tomorrow's Date : {tomorrow_date}

        {history_str}

        User Query: Timestamp in final output should be equal to {query}. 

        Instructions:
        - Resolve all relative time expressions (e.g., today, tomorrow).
        - Identify the intended future interview date and time from the conversation.
        - Apply the condition specified in the query (e.g., subtract 3 hours from the interview time).
        - Output the final result.
        - Output format: YYYY-MM-DDTHH:MM:SS (No 'Z' suffix).
        - Return ONLY the timestamp string.
        - Do NOT include any explanations or extra text.
        """
        print(f"\n--- TOOL FINAL OPENAI PROMPT (textgen_trigger_node_wait) ---\n{prompt}\n----------------------------------------------------------\n")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": default_model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are a strict timestamp extraction and calculation assistant.
        Understand the user's query and use the conversation history and information regarding current date, time and tomorrow's date to come up with the requrired output as 
        per instructions provided.
        Follow instructions EXACTLY. Do not skip steps. Do not infer missing data."""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }

        
        try:
            resp = await http_client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            
            raw_content = resp.json()["choices"][0]["message"]["content"]
            print(f"\n--- OPENAI RAW RESPONSE (textgen_trigger_node_wait) ---\n{raw_content}\n------------------------------------------------------\n")
            
            timestamp = raw_content.strip()
            
            # Clean up the response
            timestamp = timestamp.strip('`').strip('"').strip("'").strip()
            print(f"DEBUG: Extracted Timestamp: {timestamp}")
            
            import json
            return json.dumps({
                "setNextWaitUntil": timestamp,
                "instruction": f"Set setNextWaitUntil to {timestamp}"
            })
        except Exception as e:
            logger.exception(f"Error in textgen_trigger_node_wait: {e}")
            return f"Error updating wait time: {str(e)}"

    return [
        merakle_demo_get_service_request_id,
        search_knowledge,
        textgen_trigger_node_wait
    ]


# -------------------------------------------------------
# 4. FastAPI App
# -------------------------------------------------------

app = FastAPI()


class AgentRequest(BaseModel):
    accountId: Any
    campaignId: Any
    taskId: Any
    currentNode: Any
    chatHistory: List[Dict[str, Any]]
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
            # entrypoint is common in newer Agno versions
            func = getattr(t, "entrypoint", None) or getattr(t, "entry", None) or getattr(t, "function", None)
            
            if not func and callable(t):
                func = t

            if func:
                sig = inspect.signature(func)
                for name, param in sig.parameters.items():
                    # Skip 'self' or other common internal params
                    if name in ("self", "cls"):
                        continue
                    # Skip injected Agno params
                    if name in ("agent", "run_context", "team"):
                        continue
                        
                    params.append(ToolParameter(
                        name=name,
                        type=str(param.annotation.__name__) if hasattr(param.annotation, "__name__") else str(param.annotation)
                    ))
            elif hasattr(t, "parameters") and isinstance(t.parameters, dict):
                # Fallback to Agno's internal parameters schema
                props = t.parameters.get("properties", {})
                for name, details in props.items():
                    params.append(ToolParameter(
                        name=name,
                        type=details.get("type", "any")
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
        account_id = str(request.accountId)
        current_node = str(request.currentNode)
        camp_id = str(request.campaignId)
        logger.info(f"--- New Request: Task {task_id} and campaign id {camp_id}---")

        # Request-scoped tool cache
        tool_cache = {}

        # Extract settings
        ts = request.templateSettings
        campaign_settings = ts.get("campaign_settings", {})
        
        # Use LLM model from campaign settings if present, else fallback to global DEFAULT_MODEL
        default_model = ts.get("model") or campaign_settings.get("use_llm_model") or DEFAULT_MODEL
        print(f"DEBUG: Using model: {default_model}")
        
        temperature = ts.get("temperature", 0)
        
        # Check if the model is GPT-5 (prefixed with gpt-5)
        is_gpt_5 = default_model.lower().startswith("gpt-5")
        
        base_prompt = ts.get("callprompt", "You are a helpful assistant.")

        enable_tools = campaign_settings.get(
            "enable_merakle_knowledge", False
        )

        # -------------------------------------------------------
        # 5. Register Tools
        # -------------------------------------------------------

        agno_tools = []
        if enable_tools:
            logger.info("Registering MCP-backed tools")
            agno_tools = get_tools(str(request.campaignId), tool_cache, request.chatHistory, default_model)
            print(f"DEBUG: Registered tools: {agno_tools}")

        # Extract system instruction from chat history if present
        system_content = ""
        start_index = 0
        if request.chatHistory and request.chatHistory[0].get("role") == "system":
            system_content = request.chatHistory[0].get("content", "")
            start_index = 1

        # -------------------------------------------------------
        # 6. Initialize Agent
        # -------------------------------------------------------

        agent_instructions = [
            base_prompt,
            f"System Instructions: {system_content}" if system_content else None,
            "Respond naturally to the user.",
            "Use tools when necessary.",
            "If a tool fails, inform the user and move on.",
            "Do not retry a tool more than once."
        ]
        # Filter out None values
        agent_instructions = [i for i in agent_instructions if i]

        # Prepare model parameters
        main_model_params = {
            "id": default_model,
            "api_key": OPENAI_API_KEY,
        }
        if not is_gpt_5:
            main_model_params["temperature"] = temperature

        agent = Agent(
            model=OpenAIChat(**main_model_params),
            instructions=agent_instructions,
            tools=agno_tools,
            output_schema=WhatsAppResponse,
            markdown=False,
            debug_mode=True,
            tool_call_limit=5, # Increased limit to allow processing tool results.

        )

        # -------------------------------------------------------
        # 7. Format Chat History
        # -------------------------------------------------------

        chat_history_text = "\n\nConversation History:\n"

        for msg in request.chatHistory[start_index:-1]:
            role = msg.get("role", "user").title()
            chat_history_text += f"{role}: {msg.get('content')}\n"

        last_msg = request.chatHistory[-1]["content"] if request.chatHistory else "Hi"

        # Dynamic context moved here to improve prompt caching of the system instructions
        dynamic_context = (
            f"\n----------------------------\n"
            f"\n\nAdditional Context:\n"
            f"- The merakle_call_id for this call is {task_id}.\n"
            f"- The merakle_account_id for this call is {account_id}.\n"
            f"- The campaign_id for this call is {camp_id}.\n"            
            f"- The current date and time is {datetime.now()}.\n"
            f"- CURRENT NODE: {current_node}.\n"
        )

        full_prompt = chat_history_text  + f"User: {last_msg}" + dynamic_context

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
            
            # Prepare validator model parameters
            val_model_params = {
                "id": default_model,
                "api_key": OPENAI_API_KEY,
            }
            if not is_gpt_5:
                val_model_params["temperature"] = 0

            validator_agent = Agent(
                model=OpenAIChat(**val_model_params),
                debug_mode=True,
                instructions=[
                    "You are a strict output validator.",
                    "Your job is to check the Main Agent's output against the JSON CONSTRAINTS and logical rules.",
                    "1) If there are simple fixes (e.g., converting '' to null, setting arrays to []), fix them directly in `fixed_response` and set `is_valid` to true.",
                    "2) Set `is_valid` to true only if the output is correct or can be easily fixed.",
                    "3) If there are major logical errors, set `is_valid` to false and provide a detailed `critique`.",
                    "4) ALWAYS provide the best possible `fixed_response` based on the Main Agent's intent.",
                    VALIDATOR_INSTRUCTIONS
                ],
                output_schema=ValidationResult,
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
            validation_payload = [
                f"CONVERSATION HISTORY:\n{full_prompt}",
                f"TOOL RESULTS:\n{tool_results_str if tool_results_str else 'No tools were called.'}",
                f"MAIN AGENT OUTPUT TO VALIDATE:\n{main_output_str}"
            ]
            validation_payload = "\n\n".join([v for v in validation_payload if v])

            print("\n--- VALIDATION PAYLOAD ---\n")
            print(validation_payload)
            print("\n--------------------------\n")
            
            val_response = await validator_agent.arun(validation_payload)
            val_result: ValidationResult = val_response.content
            final_validated_output = response.content

            if isinstance(val_result, ValidationResult):
                if val_result.is_valid and val_result.fixed_response:
                    final_validated_output = val_result.fixed_response
                    logger.info("Validation successful (or auto-fixed).")
                    print(f"DEBUG: Final Validated Output: {final_validated_output.model_dump_json(indent=2)}")
                    break
                else:
                    current_attempt += 1
                    error_msg = val_result.critique if val_result.critique else "Output failed Pydantic validation or logic constraints."
                    logger.warning(f"Validation failed: {error_msg}. Retrying main agent...")
                    retry_prompt = full_prompt + f"\n\nCRITIQUE: Your previous response was invalid.\n{error_msg}\n\nPlease correct your output."
                    response = await agent.arun(retry_prompt)
            else:
                # Fallback if validator failed to follow schema
                current_attempt += 1
                logger.error("Validator failed to return ValidationResult schema.")
                retry_prompt = full_prompt + f"\n\nCRITIQUE: Your previous response was invalid. Please ensure you follow the JSON constraints strictly."
                response = await agent.arun(retry_prompt)

        # -------------------------------------------------------
        # 9. Return Structured Response
        # -------------------------------------------------------

        if final_validated_output:
            print(f"DEBUG: RETURNED BACK OUPUT: {final_validated_output.model_dump_json(indent=2)}")
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
