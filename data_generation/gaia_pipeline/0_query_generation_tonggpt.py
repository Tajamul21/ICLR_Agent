import argparse
import json
import os
import re
import time
import random
import string
import sys
from typing import Any, Dict, List, Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== LLM Clients ==========
# Load config first so we know which client to import
from tongagent.utils import load_config
config = load_config()

max_tokens = 2048
temperature = 1.0

USE_AZURE = (getattr(config.data_generation, "llm", "openai") == "azure")

if USE_AZURE:
    from openai import AzureOpenAI
    REGION = getattr(config.data_generation, "region", "")
    MODEL = getattr(config.data_generation, "model", "gpt-4o-mini")
    API_KEY = getattr(config.data_generation, "api_key", os.getenv("AZURE_OPENAI_API_KEY", ""))
    API_BASE = getattr(config.data_generation, "api_base", getattr(config.data_generation, "ape_base", ""))  # tolerate typo
    ENDPOINT = f"{API_BASE}/{REGION}".rstrip("/")
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-02-01",
        azure_endpoint=ENDPOINT,
    )
else:
    from openai import OpenAI
    MODEL = getattr(config.data_generation, "model", "gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== Paths & Prompts ==========
JSON_PROMPT_PATHS = [
   {
        "json": "data_generation/gaia_pipeline/prompts/query/gaia_val_metadata.jsonl",
        "prompt": "data_generation/gaia_pipeline/prompts/query/gaia_val_query_generation.prompt"
   }
]

# ========== Tool Map & Normalization ==========
_RAW_TOOL_MAP = {
    'web browser': 'ask_search_agent',
    'search engine': 'ask_search_agent',
    'calculator': 'PythonInterpreter',
    'image recognition tools': 'visualizer',
    'none': 'PythonInterpreter',
    'a web browser': 'ask_search_agent',
    'a search engine': 'ask_search_agent',
    'pdf access': 'inspect_file_as_text',
    'pdf viewer': 'inspect_file_as_text',
    'microsoft excel': 'inspect_file_as_text',
    'image recognition': 'visualizer',
    'a calculator': 'PythonInterpreter',
    'ocr': 'visualizer',
    'python': 'PythonInterpreter',
    'video recognition tools': 'visualizer',
    'microsoft excel / google sheets': 'inspect_file_as_text',
    'excel': 'inspect_file_as_text',
    'color recognition': 'visualizer',
    'excel file access': 'inspect_file_as_text',
    'access to wikipedia': 'ask_search_agent',
    'image recognition/ocr': 'visualizer',
    'a file interface': 'inspect_file_as_text',
    'a web browser.': 'ask_search_agent',
    'a search engine.': 'ask_search_agent',
    'file handling': 'inspect_file_as_text',
    'a speech-to-text tool': 'inspect_file_as_text',
    'audio capability': 'inspect_file_as_text',
    'unlambda compiler': 'inspect_file_as_text',
    'a calculator.': 'PythonInterpreter',
    'google search': 'ask_search_agent',
    'jsonld file access': 'inspect_file_as_text',
    'video parsing': 'visualizer',
    'python compiler': 'PythonInterpreter',
    'word document access': 'inspect_file_as_text',
    'tool to extract text from images': 'visualizer',
    'a word reversal tool / script': 'inspect_file_as_text',
    'counter': 'PythonInterpreter',
    'xml file access': 'inspect_file_as_text',
    'access to the internet archive, web.archive.org': 'ask_search_agent',
    'text processing/diff tool': 'inspect_file_as_text',
    'gif parsing tools': 'visualizer',
    'code/data analysis tools': 'inspect_file_as_text',
    'pdf reader': 'inspect_file_as_text',
    'markdown': 'inspect_file_as_text',
    'google translate access': 'PythonInterpreter',
    'bass note data': 'inspect_file_as_text',
    'text editor': 'inspect_file_as_text',
    'xlsx file access': 'inspect_file_as_text',
    'powerpoint viewer': 'inspect_file_as_text',
    'csv file access': 'inspect_file_as_text',
    'computer algebra system': 'inspect_file_as_text',
    'video processing software': 'visualizer',
    'audio processing software': 'inspect_file_as_text',
    'computer vision': 'visualizer',
    'google maps': 'ask_search_agent',
    'access to excel files': 'inspect_file_as_text',
    'a python ide': 'inspect_file_as_text',
    'spreadsheet editor': 'inspect_file_as_text',
    'no tools required': 'PythonInterpreter',
    'image recognition and processing tools': 'visualizer',
    'computer vision or ocr': 'visualizer',
    'c++ compiler': 'inspect_file_as_text',
    'access to google maps': 'ask_search_agent',
    'youtube player': 'ask_search_agent',
    'natural language processor': 'PythonInterpreter',
    'graph interaction tools': 'inspect_file_as_text',
    'bablyonian cuniform -> arabic legend': 'inspect_file_as_text',
    'access to youtube': 'ask_search_agent',
    'image search tools': 'ask_search_agent',
    'calculator or counting function': 'PythonInterpreter',
    'a speech-to-text audio processing tool': 'inspect_file_as_text',
    'access to academic journal websites': 'ask_search_agent',
    'pdf reader/extracter': 'inspect_file_as_text',
    "rubik's cube model": 'inspect_file_as_text',
    'wikipedia': 'ask_search_agent',
    'video capability': 'visualizer',
    'image processing tools': 'visualizer',
    'image recognition software': 'visualizer',
    'youtube': 'ask_search_agent',
}

def _strip_parens(text: str) -> str:
    while True:
        a = text.find('(')
        if a == -1: break
        b = text.find(')', a + 1)
        if b == -1: break
        text = text[:a] + text[b+1:]
    return text

def normalize_tool_name(name: str) -> str:
    name = (name or "").strip().lower()
    name = _strip_parens(name)
    name = re.sub(r'\s+', ' ', name).strip().rstrip('.')
    return name

TOOL_MAP = {normalize_tool_name(k): v for k, v in _RAW_TOOL_MAP.items()}

def map_tools(tool_list: List[str]) -> List[str]:
    mapped = []
    for t in tool_list:
        norm = normalize_tool_name(t)
        if norm in TOOL_MAP:
            mapped.append(TOOL_MAP[norm])
        # soft fallback for unknowns: skip instead of crashing
    return sorted(list(set(mapped)))

# ========== IO Utils ==========
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_prompt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_identifier(length=16):
    # avoid special chars in filenames for portability
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# ========== In-Context Pool & Sampling (Stratified) ==========
def extract_user_content_from_json(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_contents = []
    for row in rows:
        try:
            tools_raw = row.get("Annotator Metadata", {}).get("Tools", "")
            tool_list = []
            for line in (tools_raw or "").split('\n'):
                idx = line.find('.')
                token = line[idx+1:] if idx != -1 else line
                token = normalize_tool_name(token)
                if token:
                    tool_list.append(token)
            mapped = map_tools(tool_list)
            query = (row.get("Question") or "").strip()
            if query:
                user_contents.append({"query": query, "tools": mapped})
        except Exception:
            # ignore malformed rows
            pass
    return user_contents

def stratified_sample(pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Round‑robin by tool‑set signature to diversify examples."""
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for ex in pool:
        key = tuple(sorted(ex.get("tools", [])))
        buckets.setdefault(key, []).append(ex)
    keys = list(buckets.keys())
    random.shuffle(keys)
    out = []
    while len(out) < k and keys:
        for key in list(keys):
            if buckets[key]:
                out.append(buckets[key].pop())
                if len(out) >= k:
                    break
            else:
                keys.remove(key)
    # if still short (rare), top up randomly
    if len(out) < k and pool:
        needed = k - len(out)
        out.extend(random.sample(pool, min(needed, len(pool))))
    return out[:k]

# ========== Prompt Builders ==========
def prompt_with_random_examples(pool: List[Dict[str, Any]], base_prompt: str, ni: int) -> str:
    inctx = stratified_sample(pool, ni)
    inctx_json = json.dumps(inctx, indent=2, ensure_ascii=False)
    return base_prompt.replace('IN_CONTEXT_EXAMPLES', inctx_json)

def build_user_prompt(mode: int, num_queries: int) -> str:
    # mode: 0 single image no web, 1 single image web, 2 multi image no web, 3 multi image web
    parts = [f"Generate {num_queries} diverse, realistic, solvable user queries as a JSON array.",
             "Each item must include fields: {\"query\": str, \"tools\": string[], \"filename\"?: str[]}."]
    if mode in (1, 3):
        parts.append("Include scenarios that explicitly require web search.")
    if mode in (2, 3):
        parts.append("Include tasks requiring multiple images and interdependent files (e.g., PDF+XLSX).")
    parts.append("Balance between question-answering and image-generation tasks; prefer 2–6 step solutions.")
    parts.append("Ensure tool usage is necessary and implied by the query; avoid ambiguous goals.")
    parts.append("Return ONLY a JSON array; no prose.")
    return " ".join(parts)

# ========== LLM Call with Robust Retries ==========
def get_chat_response(messages, model=MODEL, temperature=0.8, max_tokens=2048, n=1, max_retries=8) -> str:
    """Exponential backoff + jitter; returns content or ''."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            if n == 1:
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
            else:
                contents = [(c.message.content or "").strip() for c in resp.choices]
                if contents and contents[0]:
                    return contents[0]
        except Exception as e:
            print(f"[warn] LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
        # backoff
        jitter = random.uniform(0, 0.5)
        time.sleep(delay + jitter)
        delay = min(delay * 2, 16)
    return ""

# ========== JSON Extraction & Early Validation ==========
_JSON_ARRAY_PATTERN = re.compile(r'\[.*\]', re.DOTALL)

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Tries to extract a JSON array from the model output.
    If multiple brackets exist, takes the largest plausible array.
    """
    # strip code fences if present
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()

    m = _JSON_ARRAY_PATTERN.search(text)
    if not m:
        # Try to coerce by finding lines that look like JSON items and wrapping them
        if text.startswith("{") and text.endswith("}"):
            text = f"[{text}]"
        else:
            # last resort: find all {...} blocks and wrap
            objs = re.findall(r'\{.*?\}', text, flags=re.DOTALL)
            if objs:
                text = "[" + ",".join(objs) + "]"
            else:
                raise ValueError("No JSON array found in output.")
    else:
        text = m.group(0)

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Top‑level JSON is not a list.")
    return data

BAD_PHRASES = {"do something interesting", "randomly", "something cool", "etc.", "and so on"}

def early_query_ok(item: Dict[str, Any]) -> bool:
    """Lightweight quality gate before saving to disk."""
    q = (item.get("query") or "").strip()
    tools = item.get("tools") or []
    if not q or len(q) < 12:
        return False
    if any(bp in q.lower() for bp in BAD_PHRASES):
        return False
    # if tools claim file processing, expect some file hint
    file_tools = {"inspect_file_as_text"}
    if any(t in file_tools for t in tools):
        if not re.search(r'\b(pdf|docx|pptx|xlsx|csv|md)\b', q.lower()):
            # allow filename field as alternative
            if not item.get("filename"):
                return False
    # enforce our tool namespace only
    valid_tools = {"ask_search_agent","PythonInterpreter","visualizer","inspect_file_as_text"}
    if not set(tools).issubset(valid_tools):
        return False
    return True

# ========== Save & Merge ==========
def queries_to_json_and_save(items: List[Dict[str, Any]], save_path: str) -> str:
    json_path = os.path.join(save_path, 'query', 'query_json')
    os.makedirs(json_path, exist_ok=True)
    out_file = os.path.join(json_path, generate_identifier() + ".json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[ok] Saved {len(items)} items -> {out_file}")
    return out_file

def merge(save_dir: str, timestamp: str):
    json_dir = os.path.join(save_dir, 'query', 'query_json')
    merge_dir = os.path.join(save_dir, 'query', 'queries_merged')
    os.makedirs(merge_dir, exist_ok=True)

    merged: List[Dict[str, Any]] = []
    if not os.path.isdir(json_dir):
        print(f"[merge] no dir {json_dir}")
        return

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged.extend(data)
                elif isinstance(data, dict):
                    merged.append(data)

    # deduplicate by (query, tuple(sorted(tools)))
    seen = set()
    deduped = []
    for it in merged:
        key = (it.get("query","").strip(), tuple(sorted(it.get("tools") or [])))
        if key not in seen and early_query_ok(it):
            seen.add(key)
            deduped.append(it)

    out_path = os.path.join(merge_dir, f"gaia_query_num{len(deduped)}_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"[merge] merged {len(deduped)} items -> {out_path}")

# ========== End-to-End Generation (batched, preloaded) ==========
def load_json_prompt(json_prompt_paths: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], str]:
    """Load once (perf win)."""
    jpath = json_prompt_paths[0]['json']
    ppath = json_prompt_paths[0]['prompt']
    rows = read_jsonl(jpath)
    base_prompt = load_prompt(ppath)
    return rows, base_prompt

def build_system_prompt(base_prompt: str, pool: List[Dict[str, Any]], ni: int) -> str:
    return prompt_with_random_examples(pool, base_prompt, ni)

def call_llm_once(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type":"text", "text": user_prompt}]}
    ]
    return get_chat_response(messages=messages, model=MODEL, temperature=temperature, max_tokens=max_tokens)

def process_one_call(system_prompt: str, user_prompt: str, save_path: str) -> int:
    raw = call_llm_once(system_prompt, user_prompt)
    if not raw:
        print("[warn] empty LLM output; skipping")
        return 0
    try:
        arr = extract_json_array(raw)
    except Exception as e:
        print(f"[warn] JSON parse failed; retrying with repair prompt: {e}")
        repair_prompt = "ONLY return a valid JSON array of objects with fields: query (str), tools (string[]), filename? (string[]). No prose."
        raw2 = call_llm_once(system_prompt, repair_prompt)
        if not raw2:
            return 0
        try:
            arr = extract_json_array(raw2)
        except Exception as e2:
            print(f"[warn] JSON parse failed again: {e2}")
            return 0

    # normalize + early filter
    cleaned = []
    for it in arr:
        q = (it.get("query") or "").strip()
        tools = it.get("tools") or []
        tools = sorted(list(set([t.strip() for t in tools if isinstance(t, str)])))
        item = {"query": q, "tools": tools}
        if "filename" in it and isinstance(it["filename"], list):
            item["filename"] = [str(x) for x in it["filename"]]
        if early_query_ok(item):
            cleaned.append(item)

    if not cleaned:
        return 0
    queries_to_json_and_save(cleaned, save_path)
    return len(cleaned)

def query_generation(args):
    # preload once
    rows, base_prompt = load_json_prompt(JSON_PROMPT_PATHS)
    pool = extract_user_content_from_json(rows)
    if not pool:
        print("[fatal] No in-context pool loaded.")
        return

    # prepare prompts
    user_prompt = build_user_prompt(args.mode, args.np)

    # build multiple system prompts with diversified in-context sets
    system_prompts = [build_system_prompt(base_prompt, pool, args.ni) for _ in range(args.ngpt)]

    total_saved = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_call, sp, user_prompt, args.save_path) for sp in system_prompts]
        for fut in as_completed(futures):
            try:
                total_saved += fut.result()
            except Exception as e:
                print(f"[warn] worker failed: {e}")

    print(f"[done] Query generation completed. Saved items: {total_saved}")

# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate queries using GAIA data (improved)')
    parser.add_argument("--number", type=int, default=3000)  # preserved (unused, for compatibility)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--save-path", type=str, default='data_generation/gaia_pipeline/save')
    parser.add_argument("--mode", type=int, default=0, help="0: single image no web, 1: single image web, 2: multi image no web, 3: multi image web")
    parser.add_argument("--output_path", type=str, default='query_generation/GAIA/gaia_queries.txt')
    parser.add_argument("--ngpt", "--num_gpt_queries", type=int, dest="ngpt", default=5000)
    parser.add_argument("--ni", "--num_incontext_examples", type=int, dest="ni", default=20)
    parser.add_argument("--np", "--query_num_per_gpt_call", type=int, dest="np", default=10)
    args = parser.parse_args()

    timestamp = args.timestamp
    args.save_path = os.path.join(args.save_path, timestamp)

    # redirect stdout to log (preserve behavior)
    os.makedirs(f'data_generation/gaia_pipeline/log/{timestamp}/', exist_ok=True)
    sys.stdout = open(f'data_generation/gaia_pipeline/log/{timestamp}/{timestamp}_0_query.log', 'a', buffering=1)

    print("GAIA-based Query GENERATION STARTED (improved):")
    query_generation(args)
    merge(args.save_path, timestamp)
