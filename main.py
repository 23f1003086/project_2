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
        
        # Make DATA_CONTENT available in code
        data_var = self.exec_globals.get('DATA_CONTENT', '')
        
        full_code = f"""from openai import OpenAI
import pandas as pd
import json
import re
import io
import requests
DATA_CONTENT = '''{data_var}'''
{code}"""
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

async def agent_solve_step(quiz_content: str, data_content: Optional[str], current_url: str, email: str, secret: str) -> dict:
    """LLM analyzes quiz AND tells us what to fetch."""
    decoded = extract_quiz_content(quiz_content)
    
    print(f"=== PHASE 1: LLM reads instructions to decide what to fetch ===", flush=True)
    
    # PHASE 1: Let LLM read the quiz and tell us what URLs to fetch
    analysis_prompt = f"""
Read these quiz instructions:
{decoded}
Answer these questions:
1. Are there any URLs, files, or data sources mentioned that need to be fetched?
2. If yes, list each URL/path that needs to be fetched (one per line)
3. If no, write "NOTHING_TO_FETCH"
Format:
URLS_TO_FETCH:
http://example.com/data.csv
/api/export.json
/data.txt
"""
    
    try:
        analysis_response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You read quiz instructions and identify what needs to be fetched. Output URLs or 'NOTHING_TO_FETCH'."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_completion_tokens=500 
        )
        
        llm_analysis = analysis_response.choices[0].message.content.strip()
        print(f"=== LLM says we need to fetch: ===", flush=True)
        print(llm_analysis, flush=True)
        
    except Exception as e:
        llm_analysis = f"ANALYSIS_FAILED: {e}"
        print(f"=== LLM analysis failed: {e} ===", flush=True)
    
    # PHASE 2: Parse LLM's response and fetch what it says
    fetched_data = ""
    
    # Extract URLs from LLM's response
    urls_to_fetch = []
    if "NOTHING_TO_FETCH" not in llm_analysis and "ANALYSIS_FAILED" not in llm_analysis:
        # Look for URLs in the response
        url_patterns = [
            r'https?://[^\s\n]+',      # http://...
            r'/[^\s\n\?]+(?:\?[^\s\n]*)?',  # /path?query
            r'[a-zA-Z0-9_\-]+\.(?:csv|json|txt|pdf|html|xml)[^\s\n]*'  # file.csv, data.json
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, llm_analysis, re.IGNORECASE)
            urls_to_fetch.extend(matches)
    
    if not urls_to_fetch:
        # SMARTER pattern matching - only capture valid URLs
        backup_patterns = [
            r'href=["\']([^"\']*scrape-data[^"\']*)["\']',  # Specific to scrape-data
            r'src=["\']([^"\']+\.(?:csv|json|txt|pdf))["\']',  # File sources
            r'(?:Scrape|Download|Get|Fetch)\s+(/[^\s\(\),]+)',  # Paths after instructions
            r'https?://[^\s"\']+',  # Full URLs
            r'/[^\s"\']+\.(?:csv|json|txt|pdf|html|js)'  # File paths
        ]
        
        for pattern in backup_patterns:
            matches = re.findall(pattern, decoded, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and match.strip():
                    clean_url = match.strip('"\'')
                    # Only add if it looks like a real URL/path
                    if (clean_url.startswith(('http://', 'https://', '/')) or 
                        any(clean_url.endswith(ext) for ext in ['.csv', '.json', '.txt', '.pdf', '.js'])):
                        urls_to_fetch.append(clean_url)
    
    # Remove duplicates
    urls_to_fetch = list(set(urls_to_fetch))
    
    print(f"=== URLs to fetch: {urls_to_fetch} ===", flush=True)
    
    # PHASE 3: Actually fetch the data
    for url in urls_to_fetch:
        try:
            # Make it a full URL
            if url.startswith('/'):
                full_url = urllib.parse.urljoin(current_url, url)
            elif '://' not in url and not url.startswith('data:'):
                # Relative path without leading slash
                full_url = urllib.parse.urljoin(current_url, '/' + url)
            else:
                full_url = url
            
            print(f"=== Fetching: {full_url} ===", flush=True)
            
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.get(full_url, timeout=20)
                if resp.status_code == 200:
                    content_type = resp.headers.get('content-type', '')
                    # Check if HTML references a JavaScript file
                    if 'text/html' in content_type and '<script' in resp.text:
                        # Look for script src references
                        js_files = re.findall(r'<script[^>]+src=["\']([^"\']+\.js)["\']', resp.text, re.IGNORECASE)
                        for js_file in js_files:
                            try:
                                js_url = urllib.parse.urljoin(full_url, js_file)
                                print(f"=== Found JS reference, fetching: {js_url} ===", flush=True)
                                js_resp = await http_client.get(js_url, timeout=20)
                                if js_resp.status_code == 200:
                                    content += f"\n\n=== JavaScript from {js_url} ===\n{js_resp.text}\n"
                            except Exception as e:
                                print(f"=== Failed to fetch JS: {e} ===", flush=True)
                    
                    # Handle different content types
                    if 'application/json' in content_type:
                
                        try:
                            data = resp.json()
                            content = json.dumps(data, indent=2)
                        except:
                            content = resp.text
                    elif 'text/csv' in content_type:
                        # Parse CSV for better readability - KEEP FULL DATA
                        try:
                            import pandas as pd
                            df = pd.read_csv(io.StringIO(resp.text))
                            content = f"CSV data: {len(df)} rows, {len(df.columns)} cols\n"
                            content += f"Columns: {list(df.columns)}\n"
                            content += f"First 10 rows:\n{df.head(10).to_string()}\n\n"
                            content += f"FULL CSV DATA:\n{resp.text}"  # ADD THIS LINE
                        except:
                            content = resp.text
                    elif 'application/javascript' in content_type or url.endswith('.js'):
                        # Special handling for JavaScript files
                        js_content = resp.text
                        content = "JavaScript file content:\n"
                        # Look for data in JS files
                       
                        # Look for arrays of numbers or strings
                        array_patterns = [
                            r'const\s+\w+\s*=\s*\[([^\]]+)\]',
                            r'let\s+\w+\s*=\s*\[([^\]]+)\]',
                            r'var\s+\w+\s*=\s*\[([^\]]+)\]',
                            r'data\s*:\s*\[([^\]]+)\]',
                        ]
                        for pattern in array_patterns:
                            matches = re.findall(pattern, js_content, re.DOTALL)
                            if matches:
                                content += f"Found data array: {matches[0][:500]}\n"
                        content += js_content
                    else:
                        content = resp.text
                    
                    fetched_data += f"\n=== Data from {full_url} ===\n{content}\n"
                    print(f"=== Success! Got {len(content)} chars ===", flush=True)
                    if 'scrape-data' in url.lower():
                        print(f"=== DEBUG: FULL SCRAPED CONTENT ===", flush=True)
                        print(content, flush=True)
                        print(f"=== END DEBUG ===", flush=True)
                else:
                    fetched_data += f"\n=== Failed to fetch {full_url}: HTTP {resp.status_code} ===\n"
        except Exception as e:
            fetched_data += f"\n=== Failed to fetch {url}: {e} ===\n"
    
    # PHASE 4: Let LLM solve the quiz with fetched data
    # FIXED PROMPT - removed the literal "FIND-THIS-IN-DATA-ABOVE" instruction
    # PHASE 4: Determine if this needs CODE execution or direct answer
    needs_calculation = any(keyword in decoded.lower() for keyword in [
        'sum', 'count', 'average', 'median', 'calculate', 'aggregate', 
        'filter', 'sort', 'analyze', 'maximum', 'minimum', 'total', 'mean'
    ])
    
    llm_output = ""
    
    if needs_calculation and fetched_data:
        print(f"=== DETECTED CALCULATION TASK - GENERATING CODE ===", flush=True)
        
        code_prompt = f"""Generate Python code to solve this task.
QUESTION:
{decoded[:500]}
DATA PREVIEW (first 1000 chars - FULL data is in DATA_CONTENT variable):
{fetched_data[:1000]}
IMPORTANT: DATA_CONTENT contains the COMPLETE data, not just the preview above.
Write Python code that:
1. Processes DATA_CONTENT to answer the question
2. Stores result in variable 'result'
3. Calls capture_output(result)
Available: pandas as pd, json, re, io
Example:
import pandas as pd
import io
df = pd.read_csv(io.StringIO(DATA_CONTENT))
result = df['value'].sum()
capture_output(result)
Return ONLY code, no markdown:"""
        
        try:
            code_response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only Python code, no markdown, no explanations."},
                    {"role": "user", "content": code_prompt}
                ],
                temperature=0.2,
                max_completion_tokens=800
            )
            
            generated_code = code_response.choices[0].message.content.strip()
            generated_code = re.sub(r'^```python\n|^```\n|```$', '', generated_code, flags=re.MULTILINE).strip()
            
            print(f"=== GENERATED CODE ===\n{generated_code}\n=== EXECUTING ===", flush=True)
            
            executor = CodeExecutor(fetched_data, AIPIPE_TOKEN)
            success = executor.execute(generated_code)
            
            if success and executor.captured_output:
                llm_output = json.dumps({
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    "answer": executor.captured_output
                })
                print(f"=== CODE SUCCESS: {executor.captured_output} ===", flush=True)
            else:
                print(f"=== CODE FAILED: {executor.captured_output} ===", flush=True)
                needs_calculation = False
        
        except Exception as e:
            print(f"=== CODE GEN FAILED: {e} ===", flush=True)
            needs_calculation = False
    
    if not needs_calculation or not llm_output:
        print(f"=== USING LLM DIRECT ANSWER ===", flush=True)
        solving_prompt = f"""Answer this quiz question by analyzing the data carefully.
QUESTION:
{decoded}
DATA AVAILABLE:
{fetched_data[:3000]}
CRITICAL INSTRUCTIONS:
1. Read the question carefully
2. Analyze the data provided above
3. Extract or calculate the ACTUAL answer
4. DO NOT use placeholder text like "YOUR_ANSWER_HERE"
5. Return the REAL answer you found in the data
Return ONLY this JSON format with the ACTUAL answer:
{{
  "email": "{email}",
  "secret": "{secret}",
  "url": "{current_url}",
  "answer": "PUT_REAL_ANSWER_HERE"
}}
Remember: The answer should be the ACTUAL value from the data, not a placeholder!"""
        
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return valid JSON with email, secret, url, answer fields."},
                    {"role": "user", "content": solving_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=500
            )
            
            llm_output = response.choices[0].message.content.strip()
            print(f"=== LLM ANSWER: ===\n{llm_output}", flush=True)
            
        except Exception as e:
            llm_output = f'{{"email":"{email}","secret":"{secret}","url":"{current_url}","answer":"ERROR: {e}"}}'
            print(f"=== LLM FAILED: {e} ===", flush=True)
    
    # Parse the output
    submission_url = extract_submission_url(decoded, current_url)
    
    # Try to parse as JSON
    try:
        # First try to find JSON in the output
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # Clean up common issues
            json_str = json_str.replace('\n', ' ').replace('  ', ' ')
            parsed = json.loads(json_str)
            
            # Ensure all required fields are present
            if not isinstance(parsed, dict):
                parsed = {"answer": str(parsed)}
            
            if 'email' not in parsed:
                parsed['email'] = email
            if 'secret' not in parsed:
                parsed['secret'] = secret
            if 'url' not in parsed:
                parsed['url'] = current_url
            if 'answer' not in parsed:
                # Try to extract answer from text
                answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', llm_output)
                if answer_match:
                    parsed['answer'] = answer_match.group(1)
                else:
                    parsed['answer'] = "NO_ANSWER_FOUND"
            
            return {
                "submission_url": submission_url,
                "calculated_answer": parsed,
                "llm_raw_output": llm_output
            }
    except json.JSONDecodeError as e:
        print(f"=== JSON parse error: {e} ===", flush=True)
        print(f"=== Raw LLM output was: {llm_output} ===", flush=True)
    except Exception as e:
        print(f"=== Other parse error: {e} ===", flush=True)
    
    # If not JSON or parse failed, create a simple response
    simple_answer = {
        "email": email,
        "secret": secret,
        "url": current_url,
        "answer": "PARSE_FAILED_" + llm_output[:50]
    }
    
    return {
        "submission_url": submission_url,
        "calculated_answer": simple_answer,
        "llm_raw_output": llm_output
    }
# ------------------ Main orchestration ------------------
async def solve_quiz_task(email: str, secret: str, url: str):
    start_time = asyncio.get_event_loop().time()
    current_url = url
    SOLVER_RESULTS[email] = {
        "status": "processing",
        "start_url": url,
        "last_url": url,
        "final_outcome": None,
        "final_answer": None,
        "question_logs": []  
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

                    # Fetch linked pages content dynamically
                    linked_pages = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', decoded, re.IGNORECASE)
                    linked_content = ""
                    for link in linked_pages:
                        full_link = urllib.parse.urljoin(current_url, link)
                        try:
                            resp = await asyncio.to_thread(requests.get, full_link, timeout=15)
                            resp.raise_for_status()
                            linked_content += resp.text + "\n"
                        except Exception:
                            continue
                    
                    # Combine current page + linked pages for LLM analysis
                    combined_content = decoded + "\n" + linked_content
                    
                    # Let LLM analyze
                    analysis_result = await agent_solve_step(combined_content, data_content, current_url, email, secret)
                    raw_answer = analysis_result.get('calculated_answer')
                    llm_output_before_submit = raw_answer  

                    # ✅ Ensure JSON-serializable answer
                    import json
                    import ast
                    
                    def safe_parse_answer(answer):
                        """Safely parse answer that could be JSON string, Python dict string, or other."""
                        if answer is None:
                            return None
                        
                        # If it's already a dict/list/number, use it
                        if isinstance(answer, (dict, list, int, float, bool)):
                            return answer
                        
                        # If it's a string, try to parse it
                        if isinstance(answer, str):
                            # Try JSON first
                            try:
                                return json.loads(answer)
                            except json.JSONDecodeError:
                                # Try Python dict syntax (like {'email': ..., 'answer': ...})
                                try:
                                    parsed = ast.literal_eval(answer)
                                    return parsed
                                except (ValueError, SyntaxError):
                                    # Keep as string
                                    return answer
                        
                        # Fallback: convert to string
                        return str(answer)
                    
                    final_answer = safe_parse_answer(raw_answer)

                    SOLVER_RESULTS[email]['final_answer'] = final_answer

                    # Save logs
                    SOLVER_RESULTS[email]['question_logs'].append({
                        "page_url": current_url,
                        "quiz_content_preview": decoded[:1500],
                        "external_data_preview": (data_content or "")[:1000],
                        "llm_raw_output": analysis_result,
                        "calculated_answer": final_answer,
                        "calculated_answer_before_submit": llm_output_before_submit
                    })

                    # Determine dynamic submission URL
                    from urllib.parse import urljoin
                    form_match = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', quiz_content, re.IGNORECASE)

                    def force_absolute(path: str) -> str:
                        if not path.startswith("/"):
                            path = "/" + path
                        return path
                    
                    if form_match:
                        action_raw = force_absolute(form_match.group(1))
                        full_submit_url = urllib.parse.urljoin(current_url, action_raw)
                    else:
                        link_match = re.search(r'href=["\']([^"\']*submit[^"\']*)["\']', quiz_content, re.IGNORECASE)
                        if link_match:
                            action_raw = force_absolute(link_match.group(1))
                            full_submit_url = urllib.parse.urljoin(current_url, action_raw)
                        else:
                            full_submit_url = urllib.parse.urljoin(current_url, "/submit")

                    if not full_submit_url:
                        SOLVER_RESULTS[email]['status'] = 'completed'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Analysis complete for {current_url}. No submission endpoint found. Answer: {final_answer}"
                        break

                    # Build submission payload - handle dict answers properly
                    print(f"=== DEBUG: Building payload from answer ===", flush=True)
                    print(f"Answer type: {type(final_answer)}", flush=True)
                    print(f"Answer value: {final_answer}", flush=True)
                    
                    if isinstance(final_answer, dict):
                        # The LLM gave us a complete JSON dict
                        payload = final_answer
                        
                        # Ensure critical fields are correct
                        if 'email' not in payload:
                            payload['email'] = email
                        if 'secret' not in payload:
                            payload['secret'] = secret
                        if 'url' not in payload:
                            payload['url'] = current_url
                        
                        # If the answer field is missing but we have 'calculated_answer', use it
                        if 'answer' not in payload and 'calculated_answer' in payload:
                            payload['answer'] = payload['calculated_answer']
                        
                        print(f"=== DEBUG: Using LLM's JSON payload ===", flush=True)
                    else:
                        # Simple answer
                        payload = {
                            "email": email,
                            "secret": secret,
                            "url": current_url,
                            "answer": final_answer
                        }
                        print(f"=== DEBUG: Using simple payload ===", flush=True)
                    
                    print(json.dumps(payload, indent=2), flush=True)

                    # Defensive size check
                    try:
                        if len(json.dumps(payload).encode('utf-8')) > 1024 * 1024:
                            SOLVER_RESULTS[email]['status'] = 'failed'
                            SOLVER_RESULTS[email]['final_outcome'] = 'Submission payload exceeds 1MB'
                            break
                    except Exception:
                        pass

                    # Submit
                    try:
                        submit_response = await http_client.post(full_submit_url, json=payload)
                        submit_response.raise_for_status()
                        submission_result = submit_response.json()
                        print("========== QUIZ RESULT ==========", flush=True)
                        print(f"ANSWER: {final_answer}", flush=True)
                        print("SUBMISSION RESPONSE:", flush=True)
                        print(json.dumps(submission_result, indent=2), flush=True)
                        print("================================", flush=True)
                    except httpx.HTTPStatusError as he:
                        SOLVER_RESULTS[email]['question_logs'].append({
                            "page_url": current_url,
                            "quiz_content_preview": decoded[:1500],
                            "external_data_preview": (data_content or "")[:1000],
                            "llm_raw_output": analysis_result,
                            "calculated_answer": final_answer,
                            "error": f"Submission HTTP error: {he.response.status_code} {he}"
                        })
                        SOLVER_RESULTS[email]['status'] = 'error'
                        SOLVER_RESULTS[email]['final_outcome'] = f"Submission HTTP error: {he.response.status_code} {he}"
                        break
                    except Exception as e:
                        SOLVER_RESULTS[email]['question_logs'].append({
                            "page_url": current_url,
                            "quiz_content_preview": decoded[:1500],
                            "external_data_preview": (data_content or "")[:1000],
                            "llm_raw_output": analysis_result,
                            "calculated_answer": final_answer,
                            "error": f"Submission error: {e}"
                        })
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
                        next_url = submission_result.get('url')
                        if next_url and next_url != current_url:
                            current_url = urllib.parse.urljoin(current_url, next_url)
                            continue
                        else:
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
    return JSONResponse(status_code=200, content=json.loads(json.dumps(result)))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
