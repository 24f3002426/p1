import os
import json
import traceback
import sys
import asyncio
import concurrent.futures
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

# Runs each execute_python call in its own thread, off the asyncio event
# loop, so a hung/slow call (e.g. a requests.get() with no response) can't
# freeze the whole bot for every other user. Threads that time out are
# abandoned (Python can't forcibly kill a thread) but the event loop stays
# free to keep serving new webhook requests.
CODE_EXEC_TIMEOUT_SECONDS = 45
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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


async def execute_python_code_with_timeout(code: str) -> str:
    """Runs execute_python_code in a worker thread and enforces a hard wall-clock
    timeout, so a stuck network call inside the model's code can't hang the
    server. On timeout, returns an error string to the model (as a tool
    result) instead of blocking forever."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, execute_python_code, code),
            timeout=CODE_EXEC_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (
            f"Error executing code: timed out after {CODE_EXEC_TIMEOUT_SECONDS}s. "
            "This usually means a network call (e.g. requests.get) hung with no "
            "timeout set, or the dataset/site is too slow to fetch this way. "
            "Try adding an explicit timeout= to any requests calls, use a "
            "different source/approach, or narrow the request (smaller date "
            "range, different endpoint, etc.)."
        )

# Tool definition schema for the LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                f"Execute Python code to process data, download datasets (using requests/pandas), "
                f"and calculate answers. Only text printed with print(...) is returned to you - "
                f"a bare expression or return value is NOT captured, so always print() any value "
                f"you need to see (e.g. print(df.head()), print(result)). Execution is "
                f"capped at {CODE_EXEC_TIMEOUT_SECONDS} seconds - always pass timeout=10 (or similar) "
                f"to any requests.get/post calls, since a hang otherwise wastes your whole budget."
            ),
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
    produces the final JSON response. Returns (answer_payload, trace) where
    trace is a list of {code, result} dicts for every tool call made, so the
    full reasoning path can be logged and inspected later.
    """
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [
            {
                "role": "system",
                "content": (
                    "You are a Data Analyst Agent. You will receive data analysis questions. "
                    "Use the execute_python tool to download data (e.g. MOSPI datasets using pandas or requests) "
                    "and compute the correct answer. "
                    "CRITICAL - NEVER FABRICATE DATA: you must never invent placeholder/mock/example data "
                    "(e.g. 'State A', 'State B', made-up numbers) and treat it as if it were real. If a real "
                    "data source cannot be fetched or found after genuine attempts, your final answer must "
                    "honestly report that (e.g. {\"error\": \"could not retrieve source data\"}), never a "
                    "guess dressed up as a result. Every number or name in your final answer must trace back "
                    "to data you actually fetched and printed in a tool call - if you did not print real "
                    "fetched values in this conversation, you do not have a real answer yet. "
                    "ALWAYS pass an explicit timeout (e.g. timeout=10) to any requests.get/post call - "
                    "a hang with no timeout wastes your whole execution budget. If a source is slow, "
                    "blocked, or unreachable, try a different real URL/approach - do not fall back to "
                    "invented data just because a fetch failed. "
                    "CRITICAL: When you have the final answer, your final message MUST be a single JSON object. "
                    "Do NOT include markdown formatting (like ```json), do not include explanations, do not include any text outside the JSON object. "
                    "The shape of the JSON object will be specified in the user's prompt. Provide ONLY the specified answer JSON."
                )
            }
        ]
    
    chat_histories[chat_id].append({"role": "user", "content": latest_message})

    max_loops = 8
    loop_count = 0
    trace = []

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
                    
                    execution_result = await execute_python_code_with_timeout(code_to_run)
                    
                    trace.append({
                        "loop": loop_count,
                        "code": code_to_run,
                        "result": str(execution_result),
                    })
                    
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
                return parsed_json, trace
            except json.JSONDecodeError:
                return {"error": "Agent did not output valid JSON", "raw_output": final_text}, trace

    return {"error": "Agent exceeded maximum tool call loops."}, trace

# ---------------------------------------------------------
# Logging and Webhooks
# ---------------------------------------------------------
def log_run(chat_id: int, user_query: str, raw_llm_output: dict, trace: list):
    """Appends a run entry as a single JSON object to run.jsonl. Includes the
    full step-by-step trace (each tool call's code + result) so the log is
    actually useful for reviewing what the agent tried, not just what it
    ended with."""
    log_entry = {
        "chat_id": chat_id,
        "user_query": user_query,
        "output": raw_llm_output,
        "trace": trace,
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
            answer_payload, trace = await solve_data_task(chat_id, user_text)
            
            # 2. Log to run.jsonl (includes full tool-call trace for debugging)
            log_run(chat_id, user_text, answer_payload, trace)
            
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
            traceback.print_exc()
            # Best-effort: let the chat know something broke, instead of
            # leaving the sender with total silence and no way to tell a
            # crash apart from "still thinking."
            try:
                async with httpx.AsyncClient() as http_client:
                    await http_client.post(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        json={"chat_id": chat_id, "text": f"Internal error: {e}"},
                    )
            except Exception:
                pass

    return {"status": "ok"}

@app.get("/run.jsonl")
async def get_logs():
    """Serves the JSONL log file publicly for the grader to download via wget."""
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="application/x-jsonlines")
    return {"error": "Log file not created yet"}, 404
