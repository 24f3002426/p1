import os
import json
import traceback
import sys
from io import StringIO
from contextlib import redirect_stdout
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HOST_URL = os.getenv("HOST_URL") # e.g., https://your-ngrok-url.app

# Load your AI Pipe token
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

# Initialize the client using AI Pipe's proxy
client = AsyncOpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openai/v1"
)

LOG_FILE = "run.jsonl"
app = FastAPI()

# Store multi-turn history. Format: { chat_id: [messages] }
chat_histories = {}

# ---------------------------------------------------------
# Tool: Python Code Execution (Data Analyst Sandboxing)
# ---------------------------------------------------------
def execute_python_code(code: str) -> str:
    """
    Executes Python code in a controlled environment and returns stdout/stderr.
    """
    stdout = StringIO()
    stderr = StringIO()
    try:
        with redirect_stdout(stdout):
            exec(code, globals(), locals())
        output = stdout.getvalue()
        return output if output else "Code executed successfully. No output."
    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        formatted_error = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        return f"Error executing code:\n{formatted_error}"

# Tool definition schema for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code to process data, download datasets (using requests/pandas), and calculate answers. Prints to stdout will be returned to you.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# ---------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------
async def solve_data_task(chat_id: int, latest_message: str) -> dict:
    """
    Manages the multi-turn agent loop. Handles tool calls until the agent 
    produces the final JSON response.
    """
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {
                "role": "system",
                "content": (
                    "You are a Data Analyst Agent. You will receive data analysis questions. "
                    "Use the execute_python tool to download data (e.g. MOSPI datasets using pandas or requests) "
                    "and compute the correct answer. "
                    "CRITICAL: When you have the final answer, your final message MUST be a single JSON object. "
                    "Do NOT include markdown formatting (like ```json), do not include explanations, do not include any text outside the JSON object. "
                    "The shape of the JSON object will be specified in the user's prompt. Provide ONLY the specified answer JSON."
                )
            }
        ]
    
    chat_histories[chat_id].append({"role": "user", "content": latest_message})

    max_loops = 5
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1
        
        # Call the LLM
        response = await client.chat.completions.create(
            model="gpt-4o", # AI Pipe maps this to the correct endpoint
            messages=chat_histories[chat_id],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1 
        )
        
        message = response.choices[0].message
        
        # If the model called a tool
        if message.tool_calls:
            chat_histories[chat_id].append(message)
            
            for tool_call in message.tool_calls:
                if tool_call.function.name == "execute_python":
                    args = json.loads(tool_call.function.arguments)
                    code_to_run = args.get("code", "")
                    
                    execution_result = execute_python_code(code_to_run)
                    
                    chat_histories[chat_id].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "execute_python",
                        "content": str(execution_result)
                    })
        else:
            # No tool calls, the model provided its final text response
            final_text = message.content.strip()
            
            # Clean up markdown if the model hallucinated it despite instructions
            if final_text.startswith("```json"):
                final_text = final_text.replace("```json", "", 1)
            if final_text.startswith("```"):
                final_text = final_text.replace("```", "", 1)
            if final_text.endswith("```"):
                final_text = final_text.rstrip("```")
                
            final_text = final_text.strip()
            
            chat_histories[chat_id].append({"role": "assistant", "content": final_text})
            
            try:
                parsed_json = json.loads(final_text)
                return parsed_json
            except json.JSONDecodeError:
                return {"error": "Agent did not output valid JSON", "raw_output": final_text}

    return {"error": "Agent exceeded maximum tool call loops."}

# ---------------------------------------------------------
# Logging and Webhooks
# ---------------------------------------------------------
def log_run(chat_id: int, user_query: str, raw_llm_output: dict):
    """Appends a run entry as a single JSON object to run.jsonl."""
    log_entry = {
        "chat_id": chat_id,
        "user_query": user_query,
        "output": raw_llm_output,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"]["text"]

        try:
            # 1. Ask the agent to solve the task
            answer_payload = await solve_data_task(chat_id, user_text)
            
            # 2. Log to run.jsonl
            log_run(chat_id, user_text, answer_payload)
            
            # 3. Construct the exact JSON structure required by the grader
            log_url = f"{HOST_URL.rstrip('/')}/run.jsonl"
            
            response_payload = {
                "answer": answer_payload, 
                "log_url": log_url
            }
            
            # 4. Reply to Telegram
            reply_text = json.dumps(response_payload)
            
            async with httpx.AsyncClient() as http_client:
                await http_client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": reply_text},
                )
                
        except Exception as e:
            print(f"Error handling webhook: {e}")
            
    return {"status": "ok"}

@app.get("/run.jsonl")
async def get_logs():
    """Serves the JSONL log file publicly for the grader to download via wget."""
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="application/x-jsonlines")
    return {"error": "Log file not created yet"}, 404
