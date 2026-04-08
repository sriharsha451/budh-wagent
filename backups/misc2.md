VALIDATOR_INSTRUCTIONS = """
OUTPUT JSON CONSTRAINTS:
  1) In output JSON, `responseText` and `responseWATemplate` are mutually exclusive props. 
     Only one can be set at a time. 
     If `responseWATemplate` is set, then `waTemplateParams` and `waTemplateContent` must also be set.
  2) Always return a structured JSON object in the following exact format:

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

        ### Field Rules:

        * "responseText" -> Use when sending a plain text reply
        * "responseWATemplate" -> Use when sending a WhatsApp template ID
        * "saveDataVariable" and "saveDataValue" -> Use together when storing user data
        * "waTemplateParams" -> Always return an array (use [] if not applicable)
        * "waTemplateContent" -> Rendered message content for the template
        * "fileAssetId" -> Use ONLY when sending a file to the user
        * "setNextWaitUntil" -> Use only when setting a future wait time (ISO 8601 UTC format)
        * "nextNode" -> The ID of the next node to transition to, if applicable
        * "quickReplyOptions" -> Use when providing buttons/options for the user to select. Always return an array (use [] if not applicable).
        * "isYesOrNoQuestion" -> true if the responseText is a question that expects a yes or no answer, otherwise false
        * "isEndOfConversation" -> true only when the conversation is fully complete, otherwise false

        ---

        ### 🚫 Mutual Exclusivity Rule (VERY IMPORTANT):

        * "responseText" and "responseWATemplate" are **mutually exclusive**
        * If "responseText" is NOT null -> "responseWATemplate" MUST be null
        * If "responseWATemplate" is NOT null -> "responseText" MUST be null
        * NEVER set both fields at the same time
        * NEVER leave both as null unless only sending a file or save_data without user message
        * **One of `responseText` or `responseWATemplate` MUST be set.** 
        * If **both** `responseText` and `responseWATemplate` are empty, **set `fileAssetId` instead**.

        ---

        ### 📁 File Handling Rule (VERY IMPORTANT):
        "fileAssetId" must be set ONLY when the AI is sending a file to the user.
        If the user sends a file:
        DO NOT copy or reuse that file ID.
        DO NOT set "fileAssetId" unless explicitly responding with a file.
        In all other cases, set "fileAssetId" to null.          

        --- 

        ### Strict Rules:

        * Always include ALL fields in every response
        * Do NOT remove or rename any fields
        * Use null for fields that are not applicable
        * "waTemplateParams" and "quickReplyOptions" must always be an array (never null)
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
    "setNextWaitUntil": null,
    "nextNode": null,
    "quickReplyOptions": [],
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
    "setNextWaitUntil": "2026-03-30T10:00:00Z",
    "nextNode": "Logic Node #1",
    "quickReplyOptions": ["Yes", "No"],
    "isEndOfConversation": false
    }

    ✅ Valid (File response - fileAssetId set when both `responseText` and `responseWATemplate` are empty):
    {
    "responseText": null,
    "responseWATemplate": null,
    "saveDataVariable": null,
    "saveDataValue": null,
    "waTemplateParams": [],
    "waTemplateContent": null,
    "fileAssetId": "file_12345abcd",
    "setNextWaitUntil": null,
    "nextNode": null,
    "quickReplyOptions": [],
    "isEndOfConversation": false
    }

    ❌ Invalid (DO NOT DO THIS):
    {
    "responseText": "Hello",
    "responseWATemplate": "template_123"
    }
"""
