import os
import httpx
import uuid
import json
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from agno.agent import Agent
from agno.tools import tool
from agno.models.openai import OpenAIChat
from agno.utils.log import logger
from datetime import datetime, timezone, timedelta
import re
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python versions < 3.9
    try:
        from pytz import timezone as PyTZZoneInfo
        def ZoneInfo(tz): return PyTZZoneInfo(tz)
    except ImportError:
        ZoneInfo = None

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

class AppointmentParams(BaseModel):
    subject: Optional[str] = Field(None, description="Subject of the appointment")
    appointment_id: Optional[str] = Field(None, description="Unique ID of the appointment, if updating/deleting")
    title: Optional[str] = Field(None, description="Title of the appointment")
    date: Optional[str] = Field(None, description="Date of the appointment (e.g. YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="Time of the appointment")
    person: Optional[str] = Field(None, description="Person involved in the appointment")
    notes: Optional[str] = Field(None, description="Additional notes or description")
    emailBody: Optional[str] = Field(None, description="Body content for email notifications related to the appointment")
    timezone: Optional[str] = Field(None, description="Timezone for the appointment (e.g. UTC, IST, America/New_York)")


class AppointmentModel(BaseModel):
    action_name: str = Field(..., description="Action to perform: create, update, or delete")
    params: AppointmentParams = Field(default_factory=AppointmentParams)


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
    emailSubject: Optional[str] = Field(None, description="Subject line for email responses, if applicable")
    appointment: Optional[AppointmentModel] = Field(None)


class ToolParameter(BaseModel):
    name: str
    type: str


class ToolInfo(BaseModel):
    tool_name: str
    description: Optional[str]
    params: List[ToolParameter]


class AvailabilityModel(BaseModel):
    summaryUtc: Optional[Any] = Field(None, description="Summary of availability in UTC")
    activeHoursUtc: Optional[Any] = Field(None, description="Active hours in UTC")
    timezone: Optional[str] = Field(None, description="User's local IANA timezone")


# -------------------------------------------------------
# 1.1. Deterministic Validation Logic
# -------------------------------------------------------

def load_tz_mapping():
    """
    Loads timezone mapping from timezones.json.
    Maps abbr, value, and text to the first IANA key in the 'utc' list.
    """
    mapping = {}
    try:
        with open("timezones.json", "r", encoding="utf-8") as f:
            tz_data = json.load(f)
            for entry in tz_data:
                iana_keys = entry.get("utc", [])
                if not iana_keys:
                    continue
                
                primary_iana = iana_keys[0]
                
                # Map abbreviation
                abbr = entry.get("abbr")
                if abbr:
                    mapping[abbr.upper()] = primary_iana
                
                # Map value
                val = entry.get("value")
                if val:
                    mapping[val.upper()] = primary_iana
                
                # Map text (description)
                text = entry.get("text")
                if text:
                    mapping[text.upper()] = primary_iana
                    
    except Exception as e:
        logger.error(f"Failed to load timezones.json: {e}")
    
    # Ensure common ones are present as fallback/override
    overrides = {
        "IST": "Asia/Kolkata",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "EST": "America/New_York",
        "EDT": "America/New_York",
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "MST": "America/Denver",
        "MDT": "America/Denver",
    }
    mapping.update(overrides)
    return mapping

# Global TZ mapping
TZ_MAPPING = load_tz_mapping()


def is_appointment_available(appointment: AppointmentModel, availability: Optional[AvailabilityModel]) -> tuple[bool, str]:
    """
    Checks if the requested appointment time is within the provided availability slots.
    """
    logger.debug(f"is_appointment_available - Appointment: {appointment.model_dump_json() if appointment else 'None'}")
    logger.debug(f"is_appointment_available - Availability: {availability.model_dump_json() if availability else 'None'}")

    if not availability or not availability.summaryUtc:
        logger.debug("No availability or summaryUtc provided. Skipping check.")
        return True, ""  # No availability info provided, skip check

    params = appointment.params
    if not params.date or not params.time:
        logger.debug(f"Missing date ({params.date}) or time ({params.time}). Skipping check.")
        return True, ""  # Missing date/time, can't check

    # Try to get timezone
    tz_str = params.timezone or "UTC"
    
    # Map common abbreviations/names to IANA keys
    iana_tz = TZ_MAPPING.get(tz_str.upper(), tz_str)
    logger.debug(f"Resolved timezone: {tz_str} -> {iana_tz}")
    
    if ZoneInfo is None:
        logger.warning("Timezone validation skipped: neither 'zoneinfo' nor 'pytz' is available.")
        return True, ""

    try:
        # Normalize time format if needed (e.g. 3 PM -> 15:00)
        time_str = params.time
        if "AM" in time_str.upper() or "PM" in time_str.upper():
            dt_parse = datetime.strptime(time_str, "%I:%M %p")
            time_str = dt_parse.strftime("%H:%M")
        
        dt_str = f"{params.date} {time_str}"
        # Parse as naive first, then attach TZ
        naive_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        aware_dt = naive_dt.replace(tzinfo=ZoneInfo(iana_tz))
        utc_dt = aware_dt.astimezone(timezone.utc)
        logger.debug(f"Appointment time: {dt_str} ({tz_str}) -> UTC: {utc_dt}")
    except Exception as e:
        logger.error(f"Timezone resolution error: {e}")
        return False, f"Invalid date/time/timezone format '{tz_str}': {str(e)}. Ensure you use valid IANA timezone names (e.g., 'Asia/Kolkata', 'America/Los_Angeles')."

    # Parse summaryUtc
    # Expected format: " - 2026-05-11 (UTC ISO Slots): 2026-05-11T03:30:00Z → 2026-05-11T11:30:00Z"
    slots = []
    
    # Handle cases where summaryUtc might be a list or other structure due to Any type
    summary_text = ""
    if isinstance(availability.summaryUtc, str):
        summary_text = availability.summaryUtc
    elif isinstance(availability.summaryUtc, list):
        summary_text = "\n".join([str(item) for item in availability.summaryUtc])
    elif isinstance(availability.summaryUtc, dict):
        # Look for common keys if it's a dict
        summary_text = str(availability.summaryUtc.get("summary") or availability.summaryUtc.get("text") or availability.summaryUtc)
    else:
        summary_text = str(availability.summaryUtc) if availability.summaryUtc else ""
    
    lines = summary_text.split('\n')
    for line in lines:
        # Match ISO timestamps
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s*→\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line)
        if match:
            start_str, end_str = match.groups()
            # replace Z with +00:00 for fromisoformat compatibility
            start_dt = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            slots.append((start_dt, end_dt))
    
    if not slots:
        logger.warning("No availability slots could be parsed from summaryUtc.")
        return True, ""
        
    logger.debug(f"Checking {utc_dt} against {len(slots)} parsed slots.")
    for start_dt, end_dt in slots:
        # Check if appointment start time is within the slot
        if start_dt <= utc_dt < end_dt:
            logger.debug(f"Match found in slot: {start_dt} to {end_dt}")
            return True, ""
            
    logger.debug(f"No match found for {utc_dt} in any slot.")
    return False, f"The requested time {params.date} at {params.time} ({tz_str}) is outside of the available UTC slots. Please check the 'availability' provided in the request context and suggest a different time to the user."


def validate_and_fix_response(response_content: Any, current_node: str = "", chat_history: Optional[List[Dict[str, Any]]] = None, protocol: str = "whatsapp", availability: Optional[AvailabilityModel] = None) -> tuple[Optional[WhatsAppResponse], bool, Optional[str]]:
    """
    Validates the WhatsAppResponse object, applies auto-fixes for common issues,
    and returns a critique if critical constraints are violated.
    """
    if not isinstance(response_content, WhatsAppResponse):
        try:
            # Try to parse if it's a dict or string
            if isinstance(response_content, str):
                response_content = WhatsAppResponse.model_validate_json(response_content)
            elif isinstance(response_content, dict):
                response_content = WhatsAppResponse.model_validate(response_content)
            else:
                return None, False, f"Expected WhatsAppResponse object, got {type(response_content).__name__}"
        except Exception as e:
            return None, False, f"Failed to parse response into WhatsAppResponse schema: {str(e)}"

    # --- 1. Auto-Fixes & Normalization ---

    # Normalize empty strings to None for specific fields
    string_fields = [
        "responseText", "responseWATemplate", "saveDataVariable",
        "saveDataValue", "waTemplateContent", "fileAssetId",
        "setNextWaitUntil", "nextNode", "emailSubject"
    ]
    for field in string_fields:
        val = getattr(response_content, field)
        if val == "":
            setattr(response_content, field, None)

    # Ensure list fields are lists
    if response_content.waTemplateParams is None:
        response_content.waTemplateParams = []
    if response_content.quickReplyOptions is None:
        response_content.quickReplyOptions = []

    # --- 2. Logical Validation ---

    critical_errors = []

    # A. Mutual Exclusivity: responseText vs responseWATemplate
    has_text = bool(response_content.responseText)
    has_template = bool(response_content.responseWATemplate)

    # if has_text and has_template:
    #     critical_errors.append("Both 'responseText' and 'responseWATemplate' are set. They are mutually exclusive. Use only one.")

    # B. Minimum Response Requirement
    has_file = bool(response_content.fileAssetId)
    has_save = bool(response_content.saveDataVariable)

    # Check for "merakle-signal-nudge-notification" in chat_history
    is_nudge = False
    if chat_history and len(chat_history) > 0:
        last_msg = chat_history[-1]
        print(f"DEBUG: last_msg: {last_msg}")
        if last_msg.get("content") == "merakle-signal-nudge-notification":
            is_nudge = True

    if (not (has_text or has_template)) and not (has_save) and not is_nudge:
        critical_errors.append("You must provide a value for 'responseText', 'responseWATemplate'. Both the params cannot be NULL or empty string at the same time. ")
        # D. Decision Node Rules
        if "Decision" in current_node:
            critical_errors.append(
                "DECISION NODE RULE VIOLATION:\n"
                "- Evaluate user input and choose EXACTLY ONE path.\n"
                "- Once a path is selected, you MUST immediately execute the TARGET NODE in the SAME turn.\n"
                "- Do NOT stop at the decision node (set 'nextNode' to the target node).\n"
                "- The decision node itself should NOT produce user-facing output; the response MUST come from the executed target node.\n"
                "- If input is unclear, use fallback path and execute fallback node immediately."
            )

    #if has_save:
    #    critical_errors.append("Data is saved. Immediately proceed to next node as per conversation flow.")

    if protocol.upper() == "EMAIL":
        if has_file and not has_text:
            critical_errors.append("You have provided 'fileAssetId' but missing 'responseText'. You must provide 'responseText' with appropriate message to communicate with the user while sending file.")
    else:
        if has_file and not (has_text or has_template):
            critical_errors.append("You have provided 'fileAssetId' but missing 'responseText' or 'responseWATemplate'. You must provide either 'responseText' or 'responseWATemplate' with appropriate message to communicate with the user while sending file.")

    # C. Template Integrity
    if has_template and not response_content.waTemplateContent:
        critical_errors.append("'responseWATemplate' is provided but 'waTemplateContent' is missing. You must provide the rendered content of the template.")

    # D. Appointment Availability Check
    if response_content.appointment and availability:
        is_available, error_msg = is_appointment_available(response_content.appointment, availability)
        if not is_available:
            critical_errors.append(error_msg)

    print(f"DEBUG: Current Node is a Decision Node: {current_node}")


    # --- 3. Final Result ---

    if critical_errors:
        critique = "The output is invalid due to the following reasons: \n- " + "\n- ".join(critical_errors)
        return response_content, False, critique

    return response_content, True, None

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

            last_msg_content = chat_history[-1]["content"] if chat_history else "Hi"
            last_msg_role = chat_history[-1].get("role", "user").title() if chat_history else "User"
            history_str += f"{last_msg_role}: {last_msg_content}"

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
            "model": "gpt-4.1-mini",
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

            return timestamp

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
    callWorkflow: Optional[Dict[str, Any]] = None
    protocol: Any
    availability: Optional[AvailabilityModel] = None


# -------------------------------------------------------
# 1.2. Static Response Generator
# -------------------------------------------------------

def generate_static_response(node_data: dict, nodes: list) -> WhatsAppResponse:
    """
    Generates a WhatsAppResponse directly from node data without LLM intervention.
    """
    placeholders = (
        node_data.get("whatsappTemplatePlaceholders")
        or node_data.get("emailTemplatePlaceholders")
        or []
    )
    params = []
    if isinstance(placeholders, list):
        # Extract the 'value' from each placeholder object
        params = [str(p.get("value", "")) for p in placeholders]

    next_node_id = node_data.get("nextNode")
    target_label = None
    if next_node_id:
        target_node = next((n for n in nodes if str(n.get("id")) == str(next_node_id)), None)
        if target_node:
            target_label = target_node.get("data", {}).get("label")

    is_end = next_node_id == "end-call" or target_label == "End Call"

    return WhatsAppResponse(
        responseText=node_data.get("messageContent") or node_data.get("whatsappTemplateContent") or node_data.get("emailTemplateContent"),
        responseWATemplate=node_data.get("whatsappTemplateId") or node_data.get("emailTemplateId"),
        saveDataVariable=node_data.get("saveToVariable"),
        waTemplateParams=params,
        waTemplateContent=node_data.get("whatsappTemplateContent") or node_data.get("emailTemplateContent"),
        fileAssetId=node_data.get("sendFileToUserAssetId"),
        nextNode=None if is_end else target_label,
        isEndOfConversation=is_end,
        emailSubject=node_data.get("emailTemplateSubject")
    )


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

        last_msg_content = request.chatHistory[-1]["content"] if request.chatHistory else "Hi"

        if last_msg_content == "merakle-signal-start-conversation-message":
            logger.info("Start conversation signal detected. Using static response generator for 'Start Call' node.")
            workflow = request.callWorkflow or {}
            nodes = workflow.get("nodes", [])
            node = next((n for n in nodes if str(n.get("data", {}).get("label")) == "Start Call"), None)
            if node:
                node_data = node.get("data", {})
                # Check if we should use the template as reference (LLM) or as a static response
                if not node_data.get("useTemplateAsReference"):
                    return generate_static_response(node_data, nodes)

        # -------------------------------------------------------
        # 4.1 Handle Workflow Node Routing
        # -------------------------------------------------------
        # if request.callWorkflow and not str(last_msg_content).lower().startswith("merakle-"):
        #     workflow = request.callWorkflow
        #     nodes = workflow.get("nodes", [])
        #     current_node_id = str(request.currentNode)
        #     node = next((n for n in nodes if str(n.get("data", {}).get("label")) == current_node_id), None)

        #     if node:
        #         node_type = node.get("type")
        #         node_data = node.get("data", {})
        #         message_type = node_data.get("messageType")

        #         # If current_node is type "decision", run the main agent (LLM)
        #         # If current_node is type "standard", check messageType
        #         if node_type == "decision":
        #             logger.info(f"Node {current_node_id} is a decision node. Proceeding to LLM.")
        #         elif node_type == "standard":
        #             if message_type == "prompt":
        #                 logger.info(f"Node {current_node_id} is a standard prompt node. Proceeding to LLM.")
        #             else:
        #                 logger.info(f"Node {current_node_id} is a standard non-prompt node. Generating static response.")
        #                 return generate_static_response(node_data, nodes)
        #         else:
        #             logger.info(f"Node {current_node_id} has type '{node_type}'. Generating static response.")
        #             return generate_static_response(node_data, nodes)

        # Request-scoped tool cache
        tool_cache = {}

        # Extract settings
        ts = request.templateSettings
        campaign_settings = ts.get("campaign_settings", {})

        protocol = str(request.protocol or ts.get("protocol") or campaign_settings.get("protocol") or "WHATSAPP")

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
        if is_gpt_5:
            main_model_params["reasoning_effort"] = "low"
        else:
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

        last_msg_content = request.chatHistory[-1]["content"] if request.chatHistory else "Hi"
        last_msg_role = request.chatHistory[-1].get("role", "user").title() if request.chatHistory else "User"

        # Dynamic context moved here to improve prompt caching of the system instructions
        now = datetime.now()
        current_day = now.strftime("%A")
        
        # Get availability timezone if provided
        avail_tz = ""
        if request.availability and request.availability.timezone:
            avail_tz = f"- The assistant/system timezone is {request.availability.timezone}.\n"
            
        dynamic_context = (
            f"\n----------------------------\n"
            f"\n\nAdditional Context:\n"
            f"- The merakle_call_id for this call is {task_id}.\n"
            f"- The merakle_account_id for this call is {account_id}.\n"
            f"- The campaign_id for this call is {camp_id}.\n"
            f"- The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')} ({current_day}).\n"
            f"{avail_tz}"
            #f"- CURRENT NODE: {current_node}.\n"
        )

        full_prompt = chat_history_text  + f"{last_msg_role}: {last_msg_content}" + dynamic_context

        print("\n--- FINAL PROMPT ---\n")
        print(full_prompt)
        print("\n--------------------\n")

        # -------------------------------------------------------
        # 8. Run Agent with Deterministic Validation Loop
        # -------------------------------------------------------

        response = await agent.arun(full_prompt)

        # Max retries for self-correction
        max_retries = 2
        current_attempt = 0

        final_validated_output = response.content

        while current_attempt < max_retries:
            logger.info(f"Running Deterministic Validator (Attempt {current_attempt + 1})...")

            # Use Python-based validation and auto-fixing
            fixed_response, is_valid, critique = validate_and_fix_response(response.content, current_node, request.chatHistory, protocol, request.availability)

            if is_valid:
                final_validated_output = fixed_response
                logger.info("Validation successful (deterministic).")
                print(f"DEBUG: Final Validated Output: {final_validated_output.model_dump_json(indent=2)}")
                break
            else:
                current_attempt += 1
                logger.warning(f"Validation failed: {critique}. Retrying main agent...")
                retry_prompt = full_prompt + f"\n\nCRITIQUE: Your previous response was invalid.\n{critique}\n\nPlease correct your output strictly according to the rules."
                response = await agent.arun(retry_prompt)
                final_validated_output = response.content

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
