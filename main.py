"""
B.L.I.T.Z. Backend — FastAPI + WebSocket Streaming + LangGraph-style Agent Swarm
Phase 1: WebSockets + Streaming + Tavily Search + Vision + Memory
"""
import os, time, json, asyncio, re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from config import config

# ── Groq ─────────────────────────────────────────────────────────────────────
from groq import AsyncGroq
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY)

# ── Tavily ───────────────────────────────────────────────────────────────────
try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=config.TAVILY_API_KEY) if config.TAVILY_API_KEY else None
except ImportError:
    tavily = None
    print("[WARN] tavily-python not installed — web search disabled")

# ── Memory (JSON file — swap for Pinecone later) ──────────────────────────────
import pathlib
MEMORY_FILE = pathlib.Path("memory.json")

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"entries": [], "context": ""}

def save_memory(data: dict):
    MEMORY_FILE.write_text(json.dumps(data, indent=2))

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics = {"requests": 0, "tokens": 0, "avg_latency_ms": 0, "errors": 0, "cpu_percent": 0, "memory_mb": 0}
_latencies: list[int] = []

app = FastAPI(title="B.L.I.T.Z. API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
class CommandRequest(BaseModel):
    text: str
    active_modules: list[str] = []
    system_override: str = ""   # Canvas / Nexus / Studio pass custom system prompts

class MemoryRequest(BaseModel):
    entry: str

class VisionRequest(BaseModel):
    image_b64: str
    question: str = "Describe this image in detail."

class ForgeRequest(BaseModel):
    code: str
    language: str = "python"
    instruction: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(active_modules: list[str], web_ctx: str = "") -> str:
    mem = load_memory()
    ctx = mem.get("context", "") or "\n".join(mem.get("entries", []))
    mode = "none"
    for m in ["coding", "studying", "create", "research", "memory", "vision", "voice"]:
        if m in active_modules:
            mode = m
            break
    if active_modules and mode == "none":
        mode = "general"
    mode_instr = config.MODE_PROMPTS.get(mode, config.MODE_PROMPTS["general"])
    if web_ctx:
        mode_instr += f"\n\nLIVE WEB RESULTS (cite naturally):\n{web_ctx}"
    return config.BASE_PROMPT.format(
        context=ctx or "No personal context saved yet.",
        active_modules=", ".join(active_modules) if active_modules else "none",
        mode=mode.upper(),
        mode_instructions=mode_instr
    )

# ─────────────────────────────────────────────────────────────────────────────
# AGENT SWARM (Planner → parallel Researchers → Critic → Writer)
# ─────────────────────────────────────────────────────────────────────────────
async def agent_planner(question: str) -> list[str]:
    r = await groq_client.chat.completions.create(
        model=config.SWARM_MODEL,
        messages=[
            {"role": "system", "content": "You are a research Planner. Given a question, output a JSON array of 2-3 specific sub-tasks needed to answer it thoroughly. Output ONLY valid JSON array of strings, no markdown, no explanation."},
            {"role": "user", "content": question}
        ],
        max_tokens=300, temperature=0.3
    )
    try:
        raw = r.choices[0].message.content.strip()
        tasks = json.loads(raw)
        return tasks if isinstance(tasks, list) else [question]
    except Exception:
        return [question]

async def agent_researcher(subtask: str, web_ctx: str) -> str:
    r = await groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": f"You are a Research Specialist. Answer this sub-task concisely and factually.\nAvailable web context:\n{web_ctx[:1500]}"},
            {"role": "user", "content": subtask}
        ],
        max_tokens=500, temperature=0.5
    )
    return r.choices[0].message.content.strip()

async def agent_critic(draft: str) -> str:
    r = await groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a Critic. In 2-3 bullet points, identify factual gaps, logical errors, or missing nuance. Be concise and precise."},
            {"role": "user", "content": draft}
        ],
        max_tokens=200, temperature=0.4
    )
    return r.choices[0].message.content.strip()

async def run_agent_swarm(question: str, web_ctx: str = "") -> tuple[str, list[str]]:
    """Planner → parallel Researchers → Critic → synthesis prompt."""
    logs = []
    try:
        logs.append("🟡 [PLANNER] Decomposing query into sub-tasks...")
        subtasks = await agent_planner(question)
        logs.append(f"✅ [PLANNER] {len(subtasks)} sub-tasks: {subtasks[:3]}")

        logs.append(f"🔵 [RESEARCHERS] Dispatching {len(subtasks)} parallel agents...")
        results = await asyncio.gather(*[agent_researcher(t, web_ctx) for t in subtasks[:3]])
        combined = "\n\n".join(f"**Sub-task: {subtasks[i]}**\n{r}" for i, r in enumerate(results))
        logs.append(f"✅ [RESEARCHERS] All {len(results)} agents returned.")

        logs.append("🔴 [CRITIC] Reviewing for gaps and errors...")
        critique = await agent_critic(combined)
        logs.append(f"✅ [CRITIC] {critique[:100]}...")

        logs.append("🟣 [WRITER] Synthesizing final answer...")
        synthesis_prompt = (
            f"Research findings from multiple agents:\n{combined}\n\n"
            f"Critic's notes to address:\n{critique}\n\n"
            f"Original question: {question}\n\n"
            f"Write a comprehensive, well-structured markdown answer that addresses the critic's notes and incorporates all findings."
        )
        return synthesis_prompt, logs
    except Exception as e:
        logs.append(f"❌ [SWARM ERROR] {e}")
        return question, logs

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET STREAMING ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            text = data.get("text", "").strip()
            active_modules = data.get("active_modules", [])
            use_swarm = data.get("use_swarm", False)
            if not text:
                continue

            t0 = time.time()
            _metrics["requests"] += 1
            web_ctx = ""

            # Web search
            if ("research" in active_modules or use_swarm) and tavily:
                try:
                    await ws.send_text(json.dumps({"type": "status", "text": "🔍 Searching the web..."}))
                    results = tavily.search(text, max_results=5, search_depth="advanced")
                    web_ctx = "\n\n".join(
                        f"[{r['title']}]({r['url']})\n{r.get('content','')[:400]}"
                        for r in results.get("results", [])
                    )
                except Exception as e:
                    web_ctx = f"(Web search error: {e})"

            # Agent swarm
            actual_prompt = text
            if use_swarm:
                actual_prompt, swarm_logs = await run_agent_swarm(text, web_ctx)
                for log in swarm_logs:
                    await ws.send_text(json.dumps({"type": "swarm_log", "text": log}))

            # Stream LLM response
            system = build_system_prompt(active_modules, web_ctx)
            full_response = ""
            try:
                stream = await groq_client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": actual_prompt}
                    ],
                    max_tokens=2048, temperature=0.7, stream=True
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_response += delta
                        await ws.send_text(json.dumps({"type": "token", "text": delta}))
            except Exception as e:
                err = f"**Error:** {e}"
                await ws.send_text(json.dumps({"type": "token", "text": err}))
                full_response = err
                _metrics["errors"] += 1

            # Done
            lat = int((time.time() - t0) * 1000)
            _latencies.append(lat)
            if len(_latencies) > 100:
                _latencies.pop(0)
            _metrics["avg_latency_ms"] = int(sum(_latencies) / len(_latencies))

            hud_data = None
            match = re.search(r"```(\w*)\n([\s\S]+?)```", full_response)
            if match:
                hud_data = {"title": match.group(1) or "Output", "content": match.group(2).strip(), "type": "CODE"}

            route = "swarm" if use_swarm else ("research" if "research" in active_modules else "llm")
            await ws.send_text(json.dumps({
                "type": "done",
                "latency_ms": lat,
                "route": route,
                "hud": hud_data
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "text": str(e)}))
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {"status": "ok", "version": "3.0.0", "name": "B.L.I.T.Z. API"}

@app.get("/")
async def root():
    index_path = _pl.Path(__file__).parent / "index.html"
    return FileResponse(index_path, media_type="text/html")

@app.get("/api/metrics")
async def metrics():
    try:
        import psutil
        _metrics["cpu_percent"] = psutil.cpu_percent()
        _metrics["memory_mb"] = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        pass
    return _metrics

@app.post("/api/connect/{module_id}")
async def connect_module(module_id: str):
    t0 = time.time()
    await asyncio.sleep(0.05)  # simulate handshake
    return {"status": "connected", "module": module_id, "latency_ms": int((time.time()-t0)*1000)}

@app.post("/api/disconnect/{module_id}")
async def disconnect_module(module_id: str):
    return {"status": "disconnected", "module": module_id}

# REST fallback for /api/command (used if WS unavailable)
@app.post("/api/command")
async def command(req: CommandRequest):
    t0 = time.time()
    _metrics["requests"] += 1
    web_ctx = ""

    if "research" in req.active_modules and tavily:
        try:
            results = tavily.search(req.text, max_results=4)
            web_ctx = "\n\n".join(
                f"[{r['title']}]\n{r.get('content','')[:350]}"
                for r in results.get("results", [])
            )
        except Exception:
            pass

    system = req.system_override if req.system_override else build_system_prompt(req.active_modules, web_ctx)
    try:
        r = await groq_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": req.text}],
            max_tokens=1500, temperature=0.7
        )
        response = r.choices[0].message.content
    except Exception as e:
        response = f"**Error:** {e}"
        _metrics["errors"] += 1

    lat = int((time.time() - t0) * 1000)
    _latencies.append(lat)
    if len(_latencies) > 100:
        _latencies.pop(0)

    hud_data = None
    match = re.search(r"```(\w*)\n([\s\S]+?)```", response)
    if match:
        hud_data = {"title": match.group(1) or "Output", "content": match.group(2).strip(), "type": "CODE"}

    mode = next((m for m in ["coding","studying","create","research","memory","vision","voice"] if m in req.active_modules), "general")
    return {
        "response": response,
        "latency_ms": lat,
        "route": mode,
        "spawn_hud": bool(hud_data),
        "hud_title": hud_data["title"] if hud_data else None,
        "hud_content": hud_data["content"] if hud_data else None,
        "hud_type": hud_data["type"] if hud_data else None,
    }

# Vision — Llama 3.2 Vision
@app.post("/api/vision")
async def vision(req: VisionRequest):
    try:
        r = await groq_client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_b64}"}},
                {"type": "text", "text": req.question}
            ]}],
            max_tokens=1000
        )
        return {"response": r.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Forge — AI code assistance
@app.post("/api/forge")
async def forge(req: ForgeRequest):
    system = """You are B.L.I.T.Z. in FORGE mode — a senior autonomous coding agent.
Given code and an instruction or error, fix/improve the code.
Always respond with:
1. A brief 1-2 sentence explanation
2. The complete corrected code in a properly fenced code block with correct language tag.
Be precise. No verbose commentary. Be like a 10x engineer."""
    user_msg = f"Language: {req.language}\nInstruction: {req.instruction}\n\nCode:\n```{req.language}\n{req.code}\n```"
    try:
        r = await groq_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            max_tokens=2000, temperature=0.25
        )
        return {"response": r.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Memory
@app.post("/api/memory")
async def save_mem(req: MemoryRequest):
    data = load_memory()
    data["entries"].append(req.entry)
    data["context"] = "\n".join(data["entries"][-30:])
    save_memory(data)
    return {"status": "saved"}

@app.get("/api/memory-view")
async def view_mem():
    data = load_memory()
    return {"content": "\n".join(data.get("entries", []))}

@app.post("/api/memory-clear")
async def clear_mem():
    save_memory({"entries": [], "context": ""})
    return {"status": "cleared"}

# Standalone web search
@app.get("/api/search")
async def web_search(q: str):
    if not tavily:
        return {"error": "Tavily not configured — add TAVILY_API_KEY to .env"}
    try:
        results = tavily.search(q, max_results=5, search_depth="advanced")
        return {"results": results.get("results", [])}
    except Exception as e:
        return {"error": str(e)}

# Web scrape for Nexus (returns extracted text for blocked iframes)
class ScrapeRequest(BaseModel):
    url: str

@app.post("/api/scrape")
async def scrape(req: ScrapeRequest):
    """Fetch a URL and return its readable text content for Nexus windows."""
    import httpx
    if not req.url.startswith("http"):
        return JSONResponse(status_code=400, content={"error": "Invalid URL"})
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BLITZ/3.0)"}) as client:
            r = await client.get(req.url)
            html = r.text
        # Light extraction — strip tags, collapse whitespace
        import re as _re
        # Remove scripts, styles, nav, footer cruft
        html = _re.sub(r"<(script|style|nav|footer|header)[^>]*>[\s\S]*?</\1>", "", html, flags=_re.I)
        # Extract title
        title_m = _re.search(r"<title[^>]*>(.*?)</title>", html, _re.I | _re.S)
        title = title_m.group(1).strip() if title_m else req.url
        # Strip all tags
        text = _re.sub(r"<[^>]+>", " ", html)
        text = _re.sub(r"&nbsp;", " ", text)
        text = _re.sub(r"&amp;", "&", text)
        text = _re.sub(r"&lt;", "<", text)
        text = _re.sub(r"&gt;", ">", text)
        text = _re.sub(r"\s{3,}", "\n\n", text).strip()
        return {"title": title, "content": text[:6000], "url": req.url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Serve frontend HTML files — so all pages work from same origin on Render
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib as _pl

_HTML_DIR = _pl.Path(__file__).parent  # same folder as main.py

_HTML_FILES = [
    "blitz_hub.html", "forge.html", "canvas.html",
    "nexus.html", "studio.html", "index.html",
]

for _fname in _HTML_FILES:
    _path = _HTML_DIR / _fname
    if _path.exists():
        @app.get(f"/{_fname}", include_in_schema=False)
        async def _serve(f=_path):
            return FileResponse(f, media_type="text/html")

# Also serve /hub, /forge, /canvas, /nexus, /studio as clean URLs
_CLEAN_ROUTES = {
    "hub":    "blitz_hub.html",
    "forge":  "forge.html",
    "canvas": "canvas.html",
    "nexus":  "nexus.html",
    "studio": "studio.html",
}

for _route, _file in _CLEAN_ROUTES.items():
    _fpath = _HTML_DIR / _file
    if _fpath.exists():
        @app.get(f"/{_route}", include_in_schema=False)
        async def _serve_clean(f=_fpath):
            return FileResponse(f, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
