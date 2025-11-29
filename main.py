#!/usr/bin/env python3
"""
LLM Quiz Solver

"""

import os
import json
import asyncio
import re
import io
import base64
import urllib.parse
import traceback
from typing import Optional

import requests
import httpx
from playwright.async_api import async_playwright
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Data processing
import pandas as pd
from pdfminer.high_level import extract_text_to_fp




STUDENT_SECRET = os.getenv("STUDENT_SECRET")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
QUIZ_TIMEOUT_SECONDS = int(os.getenv("QUIZ_TIMEOUT_SECONDS", "175"))

if not STUDENT_SECRET or not AIPIPE_TOKEN:
    raise ValueError("STUDENT_SECRET or AIPIPE_TOKEN not set in .env")

client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openai/v1")
http_client = httpx.AsyncClient(timeout=30)

# Global results
SOLVER_RESULTS = {}

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

class EndpointResponse(BaseModel):
    status: str
    message: str

# ------------------ Utilities ------------------

def extract_quiz_content(html_content: str) -> str:
    """Decode base64 embedded content if present (atob in script) else return raw."""
    try:
        base64_pattern = r'atob\(["\']([^"\']+)["\']\)'
        base64_match = re.search(base64_pattern, html_content)
        if base64_match:
            encoded = base64_match.group(1)
            try:
                decoded = base64.b64decode(encoded).decode("utf-8")
                return decoded
            except Exception:
                # Try urlsafe
                decoded = base64.urlsafe_b64decode(encoded + '==').decode('utf-8')
                return decoded
        return html_content
    except Exception:
        return html_content


def extract_submission_url(quiz_content: str, current_url: Optional[str]) -> Optional[str]:
    """
    Extract the true submission URL from the quiz page content.
    Always prefer URLs that explicitly appear as submission endpoints.
    Never return the quiz page URL itself.
    """

    # Strict patterns: look for a POST/submit instruction
    patterns = [
        r'POST\s+to\s+(https?://[^\s"\'<>]+)',
        r'Submit\s+(?:at|to)\s+(https?://[^\s"\'<>]+)',
        r'send\s+(?:result|answer)\s+to\s+(https?://[^\s"\'<>]+)',
        r'Submission\s+URL[:\s]*(https?://[^\s"\'<>]+)',
    ]

    # STRICT MATCHES
    for pat in patterns:
        for m in re.findall(pat, quiz_content, re.IGNORECASE):
            url = m.strip('"\' ,.)')
            full = urllib.parse.urljoin(current_url or "", url)
            return full

    # Look specifically for URLs ending with /submit
    submit_candidates = re.findall(r'https?://[^\s"\'<>]+', quiz_content)
    for url in submit_candidates:
        if url.rstrip('/').endswith('submit'):
            return urllib.parse.urljoin(current_url or "", url)

    # <form action="...">
    form_match = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', quiz_content, re.IGNORECASE)
    if form_match:
        return urllib.parse.urljoin(current_url or "", form_match.group(1))

    return None


def download_and_extract_data(url: str) -> str:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type','')
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            fp = io.BytesIO(resp.content)
            out = io.StringIO()
            extract_text_to_fp(fp, out)
            return out.getvalue()
        if 'text/csv' in content_type or url.lower().endswith(('.csv', '.tsv')):
            df = pd.read_csv(io.StringIO(resp.text))
            return df.to_markdown(index=False)
        if 'application/json' in content_type or 'text/plain' in content_type:
            return resp.text
        if any(ext in content_type.lower() for ext in ['audio','image','video']) or any(url.lower().endswith(ext) for ext in ['.mp3','.wav','.ogg','.flac','.jpg','.png','.mp4']):
            return f"MEDIA_FILE_URL: {url} | Content-Type: {content_type}"
        return f"Data downloaded, but type {content_type} unsupported for direct extraction."
    except Exception as e:
        return f"Error downloading or processing file at {url}: {e}"

# ------------------ Code executor ------------------
class CodeExecutor:
    def __init__(self, data_content: Optional[str], token: str):
        self.captured_output = ""
        self.captured_image_b64 = None
        self.exec_globals = {
            'pd': pd,
            'json': json,
            're': re,
            'io': io,
            'requests': requests,
            'OpenAI': OpenAI,
            'AIPIPE_TOKEN': token,
            'DATA_CONTENT': data_content,
            'capture_output': self._capture_output,
            'capture_chart': self._capture_chart_b64,
        }

    def _capture_output(self, result):
        self.captured_output = str(result)

    def _capture_chart_b64(self):
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        image_bytes = buf.getvalue()
        self.captured_image_b64 = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, code: str) -> bool:
        self.captured_output = ""
        self.captured_image_b64 = None
        full_code = f"from openai import OpenAI\nimport pandas as pd\nimport json\nimport re\nimport io\nimport requests\n\n{code}"
        try:
            if 'plt.' in code or 'capture_chart' in code:
                import matplotlib.pyplot as plt
                self.exec_globals['plt'] = plt
            exec(full_code, self.exec_globals)
            return True
        except Exception as e:
            self.captured_output = f"Code Execution Error: {e.__class__.__name__}: {e}"
            return False

# ------------------ JSON extraction helpers ------------------

def find_outermost_json(text: str) -> Optional[str]:
    """Return the first JSON object-like substring that parses cleanly.
    We try to find the largest balanced braces block that decodes to JSON.
    """
    # Find all brace blocks and prefer the longest that parses
    brace_spans = []
    stack = []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            start = stack.pop()
            brace_spans.append((start, i+1))
    # Sort spans by length desc so we try largest first
    brace_spans.sort(key=lambda s: s[1]-s[0], reverse=True)
    for start, end in brace_spans:
        candidate = text[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return candidate
        except Exception:
            continue
    return None

# ------------------ LLM agent step ------------------
SYSTEM_PROMPT = """
You are an expert quiz-solving AI. Generate Python code in a ```python``` block that ends with capture_output(final_result).
After the code, output ONLY a JSON object with keys submission_url and calculated_answer (these can be placeholders). Do not output any extra commentary.
"""

async def agent_solve_step(quiz_content: str, data_content: Optional[str], current_url: str) -> dict:
    decoded = extract_quiz_content(quiz_content)
    data_hint = f"DATA_CONTENT = '''{(data_content or '')[:1500]}...'''" if data_content else "No external data link found."
    user_prompt = f"""
--- QUIZ PAGE CONTENT ---
{decoded}

--- AVAILABLE DATA ---
{data_hint}

INSTRUCTIONS:
1. Produce Python code in a ```python``` block that computes the answer and ends with capture_output(final_result)
2. After the code, output a JSON object (as the only remaining text) with submission_url and calculated_answer (these may be placeholders)
"""
    # call LLM (run in thread to avoid blocking)
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            temperature=0.1
        )
    except Exception as e:
        return {"submission_url": None, "calculated_answer": f"LLM call failed: {e}"}

    llm_output = response.choices[0].message.content

    # Extract python code block if present
    code_block = None
    m = re.search(r'```python\n(.*?)```', llm_output, re.DOTALL)
    if m:
        code_block = m.group(1).strip()

    executor = CodeExecutor(data_content, AIPIPE_TOKEN)
    exec_result = ""
    if code_block:
        success = await asyncio.to_thread(executor.execute, code_block)
        exec_result = executor.captured_image_b64 or executor.captured_output or ""
    else:
        exec_result = "No executable code generated"

    # Robust JSON extraction: prefer page-extracted URL and override answers
    structured_output = {}
    json_text = find_outermost_json(llm_output)
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                structured_output.update(parsed)
        except Exception:
            pass

    # Fallback: attempt to find any explicit key:value pairs heuristically
    # 
    if not structured_output.get('submission_url'):
        # try simple pattern
        m2 = re.search(r'"submission_url"\s*:\s*"([^"]+)"', llm_output)
        if m2:
            structured_output['submission_url'] = m2.group(1)

    # IMPORTANT: enforce that the calculated_answer is ALWAYS the executor's result
    structured_output['calculated_answer'] = exec_result

    # Final best-effort: resolve submission_url from page content if available
    page_url = extract_submission_url(decoded, current_url)
    if page_url:
        # prefer page-specified endpoint
        structured_output['submission_url'] = page_url
    # if still missing, leave whatever LLM provided

    return structured_output

# ------------------ Main orchestration ------------------
async def solve_quiz_task(email: str, secret: str, url: str):
    start_time = asyncio.get_event_loop().time()
    current_url = url
    SOLVER_RESULTS[email] = {
        "status": "processing",
        "start_url": url,
        "last_url": url,
        "final_outcome": None,
        "final_answer": None
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                while current_url:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > QUIZ_TIMEOUT_SECONDS:
                        SOLVER_RESULTS[email]['status'] = 'timed_out'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Timed out after {QUIZ_TIMEOUT_SECONDS} seconds"
                        break

                    SOLVER_RESULTS[email]['status'] = f"processing | working on {current_url}"

                    page = await browser.new_page()
                    try:
                        await page.goto(current_url, wait_until='networkidle', timeout=30000)
                        quiz_content = await page.content()
                    except Exception as e:
                        SOLVER_RESULTS[email]['status'] = 'error'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Navigation error: {e}"
                        break

                    decoded = extract_quiz_content(quiz_content)

                    # Try to find data link (audio/pdf/csv)
                    data_content = None
                    data_url = None
                    audio_match = re.search(r'(?:src|href)=["\'](https?:\\/\\/[^"\']+\\.(?:mp3|wav|ogg|flac)(?:[\?#][^"\']*)?)["\']', quiz_content, re.IGNORECASE)
                    if audio_match:
                        data_url = audio_match.group(1)
                    else:
                        link_match = re.search(r'<a[^>]+href=["\'](https?://[^"\']+)["\']', quiz_content, re.IGNORECASE)
                        if link_match:
                            data_url = link_match.group(1)
                    if data_url:
                        data_content = await asyncio.to_thread(download_and_extract_data, data_url)

                    analysis_result = await agent_solve_step(quiz_content, data_content, current_url)

                    raw_answer = analysis_result.get('calculated_answer')
                    final_answer = str(raw_answer) if raw_answer is not None else ""
                    SOLVER_RESULTS[email]['final_answer'] = final_answer

                    # ALWAYS prefer URL extracted from the page
                    page_submit = extract_submission_url(decoded, current_url)

                    if page_submit:
                        full_submit_url = page_submit
                    else:
                        # fallback ONLY if page does not define submission URL
                        llm_submit = analysis_result.get('submission_url')
                        full_submit_url = urllib.parse.urljoin(current_url, llm_submit) if llm_submit else None

                    if not full_submit_url:
                        SOLVER_RESULTS[email]['status'] = 'completed'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Analysis complete for {current_url}. No submission endpoint found. Answer: {final_answer}"
                        break

                    # Build submission payload
                    payload = {
                        "email": email,
                        "secret": secret,
                        "url": current_url,
                        "answer": final_answer
                    }

                    # Defensive size check (must be < 1MB as spec)
                    try:
                        if len(json.dumps(payload).encode('utf-8')) > 1024 * 1024:
                            SOLVER_RESULTS[email]['status'] = 'failed'
                            SOLVER_RESULTS[email]['final_outcome'] = 'Submission payload exceeds 1MB'
                            break
                    except Exception:
                        pass

                    try:
                        submit_response = await http_client.post(full_submit_url, json=payload)
                        submit_response.raise_for_status()
                        submission_result = submit_response.json()
                    except httpx.HTTPStatusError as he:
                        # If we get 4xx/5xx, record and stop
                        SOLVER_RESULTS[email]['status'] = 'error'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Submission HTTP error: {he.response.status_code} {he} | {str(he)}"

                        break
                    except Exception as e:
                        SOLVER_RESULTS[email]['status'] = 'error'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Submission error: {e}"
                        break

                    correct = submission_result.get('correct')
                    SOLVER_RESULTS[email]['last_url'] = current_url

                    if correct:
                        next_url = submission_result.get('url')
                        if not next_url:
                            SOLVER_RESULTS[email]['status'] = 'completed'
                            SOLVER_RESULTS[email]['final_outcome'] = 'Quiz chain successfully solved and submitted.'
                            break
                        else:
                            current_url = urllib.parse.urljoin(current_url, next_url)
                            continue
                    else:
                        # Incorrect answer: either retry same endpoint (if url returned) or fail
                        next_url = submission_result.get('url')
                        if next_url and next_url != current_url:
                            current_url = urllib.parse.urljoin(current_url, next_url)
                            continue
                        else:
                            # The evaluator permits re-submission; but for now stop and record
                            SOLVER_RESULTS[email]['status'] = 'failed'
                            SOLVER_RESULTS[email]['final_outcome'] = f"Incorrect answer attempted: {final_answer}"
                            break

            finally:
                await browser.close()

    except Exception as gen_err:
        SOLVER_RESULTS[email]['status'] = 'fatal_error'
        SOLVER_RESULTS[email]['final_outcome'] = f"Fatal setup error: {gen_err}\n{traceback.format_exc()}"
        SOLVER_RESULTS[email]['final_answer'] = str(gen_err)

# ------------------ FastAPI endpoints ------------------
app = FastAPI(title="LLM Analysis Quiz Solver - Patched")
@app.get("/")
async def root():
    return {"message": "LLM Quiz Solver is running! Use /solve POST to submit a quiz."}
@app.post('/solve')
async def handle_quiz_request(task: QuizTask, background_tasks: BackgroundTasks):
    if task.secret != STUDENT_SECRET:
        return JSONResponse(status_code=403, content={"message":"Forbidden","detail":"Invalid secret provided."})

    current_status = SOLVER_RESULTS.get(task.email, {}).get('status')
    if current_status and current_status.startswith('processing'):
        return JSONResponse(status_code=202, content=EndpointResponse(status='processing', message=f"Agent solver is already processing a task for {task.email}. Check /result/{task.email} for status.").model_dump())

    background_tasks.add_task(solve_quiz_task, task.email, task.secret, task.url)
    return JSONResponse(status_code=200, content=EndpointResponse(status='success', message=f"Agent solver started for URL: {task.url}. Check /result/{task.email} for outcome.").model_dump())

@app.get('/result/{email}')
async def get_solver_result(email: str):
    result = SOLVER_RESULTS.get(email)
    if not result:
        raise HTTPException(status_code=404, detail=f"No quiz task found for email: {email}")
    return JSONResponse(status_code=200, content=result)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
