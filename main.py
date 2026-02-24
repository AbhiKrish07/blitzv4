"""
B.L.I.T.Z. Backend — Jarvis-Level Autonomous AI Assistant
FastAPI + WebSocket Streaming + Agent Swarm + Intent Detection + System Control
"""
import os, time, json, asyncio, re, pathlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_FILE = pathlib.Path("memory.json")

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return {"entries": [], "context": "", "preferences": {}}

def save_memory(data: dict):
    MEMORY_FILE.write_text(json.dumps(data, indent=2))

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics = {"requests": 0, "tokens": 0, "avg_latency_ms": 0, "errors": 0}
_latencies: list[int] = []
_conversation_history: list[dict] = []
MAX_HISTORY = 20

def add_to_history(role: str, content: str):
    _conversation_history.append({"role": role, "content": content})
    if len(_conversation_history) > MAX_HISTORY * 2:
        _conversation_history.pop(0)

app = FastAPI(title="B.L.I.T.Z. API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    from system_control import router as sys_router
    app.include_router(sys_router)
    print("[OK] system_control router loaded")
except Exception as _e:
    print(f"[WARN] system_control not loaded: {_e}")

# ── Models ─────────────────────────────────────────────────────────────────────
class CommandRequest(BaseModel):
    text: str
    active_modules: list[str] = []
    system_override: str = ""
    use_history: bool = True

class MemoryRequest(BaseModel):
    entry: str
    key: str = ""

class VisionRequest(BaseModel):
    image_b64: str
    question: str = "Describe this image in detail. Identify text, UI elements, errors, or actionable items."

class ForgeRequest(BaseModel):
    code: str
    language: str = "python"
    instruction: str = ""

class IntentRequest(BaseModel):
    text: str

class ScrapeRequest(BaseModel):
    url: str

class AutoRequest(BaseModel):
    text: str
    context: str = ""

# ── Intent Detection ──────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "open_app":       re.compile(r"\b(open|launch|start|run)\b.*(app|chrome|firefox|notepad|vscode|spotify|discord|slack|terminal|explorer|calculator|paint|word|excel)\b", re.I),
    "web_search":     re.compile(r"\b(search|google|look up|what is|who is|when did|how to|latest|news|weather)\b", re.I),
    "set_volume":     re.compile(r"\b(volume|sound|mute|unmute|louder|quieter|audio)\b", re.I),
    "set_brightness": re.compile(r"\b(brightness|dim|brighter|screen light)\b", re.I),
    "play_music":     re.compile(r"\b(play music|pause music|skip|next track|previous track|stop music)\b", re.I),
    "system_info":    re.compile(r"\b(cpu|ram|memory usage|battery|processes|system info)\b", re.I),
    "take_screenshot":re.compile(r"\b(screenshot|screen capture|what.*screen)\b", re.I),
    "remember":       re.compile(r"\b(remember|note that|save this|store|don.t forget)\b", re.I),
    "recall":         re.compile(r"\b(recall|what did i say|do you remember|what.*stored|my.*note)\b", re.I),
}

def detect_intent(text: str) -> str:
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(text):
            return intent
    return "chat"

# ── System Prompt Builder ─────────────────────────────────────────────────────
def build_system_prompt(active_modules: list[str], web_ctx: str = "", intent: str = "") -> str:
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

# ── Autonomous Action Handler ─────────────────────────────────────────────────
async def handle_autonomous_action(intent: str, text: str) -> dict | None:
    import httpx
    base = "http://localhost:8000"
    try:
        if intent == "take_screenshot":
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{base}/api/sys/screenshot")
                if r.status_code == 200:
                    return {"type": "screenshot", "data": r.json()}

        elif intent == "system_info":
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{base}/api/sys/info")
                if r.status_code == 200:
                    return {"type": "system_info", "data": r.json()}

        elif intent == "open_app":
            match = re.search(r"\b(open|launch|start)\b\s+(.+?)(?:\s+(?:for|please|now))?\.?$", text, re.I)
            if match:
                app_name = match.group(2).strip()
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.post(f"{base}/api/sys/open-app", json={"name": app_name})
                    if r.status_code == 200:
                        return {"type": "app_launched", "data": r.json()}

        elif intent == "set_volume":
            vol_match = re.search(r"\b(\d+)\s*%?\b", text)
            if vol_match:
                level = min(100, max(0, int(vol_match.group(1))))
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.post(f"{base}/api/sys/volume", json={"level": level})
                    if r.status_code == 200:
                        return {"type": "volume_set", "data": r.json()}
            elif re.search(r"\bmute\b", text, re.I):
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.post(f"{base}/api/sys/volume", json={"mute": True})
                    if r.status_code == 200:
                        return {"type": "muted", "data": r.json()}

        elif intent == "set_brightness":
            bright_match = re.search(r"\b(\d+)\s*%?\b", text)
            if bright_match:
                level = min(100, max(1, int(bright_match.group(1))))
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.post(f"{base}/api/sys/brightness", json={"level": level})
                    if r.status_code == 200:
                        return {"type": "brightness_set", "data": r.json()}

        elif intent == "play_music":
            action_map = {"pause": "pause", "stop": "pause", "next": "next", "previous": "previous", "play": "play", "skip": "next"}
            for word, action in action_map.items():
                if re.search(rf"\b{word}\b", text, re.I):
                    async with httpx.AsyncClient(timeout=5) as c:
                        r = await c.post(f"{base}/api/sys/spotify/control", json={"action": action})
                        if r.status_code == 200:
                            return {"type": "music_control", "data": r.json()}
                    break

    except Exception as e:
        print(f"[AUTO ACTION ERROR] {intent}: {e}")
    return None

# ── Agent Swarm ────────────────────────────────────────────────────────────────
async def agent_planner(question: str) -> list[str]:
    r = await groq_client.chat.completions.create(
        model=config.SWARM_MODEL,
        messages=[
            {"role": "system", "content": "Output a JSON array of 2-3 specific research sub-tasks for the question. JSON array only, no markdown."},
            {"role": "user", "content": question}
        ],
        max_tokens=300, temperature=0.3
    )
    try:
        tasks = json.loads(r.choices[0].message.content.strip())
        return tasks if isinstance(tasks, list) else [question]
    except Exception:
        return [question]

async def agent_researcher(subtask: str, web_ctx: str) -> str:
    r = await groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": f"Research Specialist. Answer concisely.\nWeb context:\n{web_ctx[:1500]}"},
            {"role": "user", "content": subtask}
        ],
        max_tokens=500, temperature=0.5
    )
    return r.choices[0].message.content.strip()

async def agent_critic(draft: str) -> str:
    r = await groq_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": "Critic. In 2-3 bullets, identify factual gaps or errors. Be precise."},
            {"role": "user", "content": draft}
        ],
        max_tokens=200, temperature=0.4
    )
    return r.choices[0].message.content.strip()

async def run_agent_swarm(question: str, web_ctx: str = "") -> tuple[str, list[str]]:
    logs = []
    try:
        logs.append("🟡 [PLANNER] Decomposing query...")
        subtasks = await agent_planner(question)
        logs.append(f"✅ [PLANNER] {len(subtasks)} sub-tasks")

        logs.append(f"🔵 [RESEARCHERS] {len(subtasks)} parallel agents...")
        results = await asyncio.gather(*[agent_researcher(t, web_ctx) for t in subtasks[:3]])
        combined = "\n\n".join(f"**{subtasks[i]}**\n{r}" for i, r in enumerate(results))
        logs.append("✅ [RESEARCHERS] All returned.")

        logs.append("🔴 [CRITIC] Reviewing...")
        critique = await agent_critic(combined)
        logs.append(f"✅ [CRITIC] Done")

        logs.append("🟣 [WRITER] Synthesizing...")
        synthesis = (
            f"Research findings:\n{combined}\n\nCritic notes:\n{critique}\n\n"
            f"Question: {question}\n\nWrite a comprehensive markdown answer addressing all findings."
        )
        return synthesis, logs
    except Exception as e:
        logs.append(f"❌ [SWARM ERROR] {e}")
        return question, logs

# ── WebSocket Streaming ────────────────────────────────────────────────────────
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
            use_history = data.get("use_history", True)
            if not text:
                continue

            t0 = time.time()
            _metrics["requests"] += 1
            web_ctx = ""

            intent = detect_intent(text)
            await ws.send_text(json.dumps({"type": "intent", "intent": intent}))

            # Autonomous actions
            if intent not in ("chat", "web_search", "remember", "recall"):
                action_result = await handle_autonomous_action(intent, text)
                if action_result:
                    await ws.send_text(json.dumps({"type": "action_result", "data": action_result}))

            # Web search
            if ("research" in active_modules or use_swarm or intent == "web_search") and tavily:
                try:
                    await ws.send_text(json.dumps({"type": "status", "text": "🔍 Searching the web..."}))
                    results = tavily.search(text, max_results=5, search_depth="advanced")
                    web_ctx = "\n\n".join(
                        f"[{r['title']}]({r['url']})\n{r.get('content','')[:400]}"
                        for r in results.get("results", [])
                    )
                except Exception as e:
                    web_ctx = f"(Web search error: {e})"

            actual_prompt = text
            if use_swarm:
                actual_prompt, swarm_logs = await run_agent_swarm(text, web_ctx)
                for log in swarm_logs:
                    await ws.send_text(json.dumps({"type": "swarm_log", "text": log}))

            system = build_system_prompt(active_modules, web_ctx, intent)
            messages = [{"role": "system", "content": system}]
            if use_history and _conversation_history:
                messages.extend(_conversation_history[-MAX_HISTORY:])
            messages.append({"role": "user", "content": actual_prompt})

            full_response = ""
            try:
                stream = await groq_client.chat.completions.create(
                    model=config.MODEL_NAME,
                    messages=messages,
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

            add_to_history("user", text)
            add_to_history("assistant", full_response)

            if intent == "remember":
                mem = load_memory()
                mem["entries"].append(f"[{time.strftime('%Y-%m-%d')}] {text}")
                mem["context"] = "\n".join(mem["entries"][-30:])
                save_memory(mem)

            lat = int((time.time() - t0) * 1000)
            _latencies.append(lat)
            if len(_latencies) > 100:
                _latencies.pop(0)
            _metrics["avg_latency_ms"] = int(sum(_latencies) / len(_latencies))

            hud_data = None
            match = re.search(r"```(\w*)\n([\s\S]+?)```", full_response)
            if match:
                hud_data = {"title": match.group(1) or "Output", "content": match.group(2).strip(), "type": "CODE"}

            route = "swarm" if use_swarm else (intent if intent != "chat" else "llm")
            await ws.send_text(json.dumps({
                "type": "done",
                "latency_ms": lat,
                "route": route,
                "intent": intent,
                "hud": hud_data
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "text": str(e)}))
        except Exception:
            pass

# ── REST Endpoints ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {"status": "ok", "version": "4.0.0", "name": "B.L.I.T.Z. API", "model": config.MODEL_NAME}

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
    await asyncio.sleep(0.05)
    return {"status": "connected", "module": module_id, "latency_ms": int((time.time()-t0)*1000)}

@app.post("/api/disconnect/{module_id}")
async def disconnect_module(module_id: str):
    return {"status": "disconnected", "module": module_id}

@app.post("/api/intent")
async def detect_intent_api(req: IntentRequest):
    return {"intent": detect_intent(req.text), "text": req.text}

@app.post("/api/command")
async def command(req: CommandRequest):
    t0 = time.time()
    _metrics["requests"] += 1
    web_ctx = ""
    intent = detect_intent(req.text)

    if "research" in req.active_modules and tavily:
        try:
            results = tavily.search(req.text, max_results=4)
            web_ctx = "\n\n".join(f"[{r['title']}]\n{r.get('content','')[:350]}" for r in results.get("results", []))
        except Exception:
            pass

    system = req.system_override if req.system_override else build_system_prompt(req.active_modules, web_ctx, intent)
    messages = [{"role": "system", "content": system}]
    if req.use_history and _conversation_history:
        messages.extend(_conversation_history[-10:])
    messages.append({"role": "user", "content": req.text})

    try:
        r = await groq_client.chat.completions.create(
            model=config.MODEL_NAME, messages=messages, max_tokens=1500, temperature=0.7
        )
        response = r.choices[0].message.content
    except Exception as e:
        response = f"**Error:** {e}"
        _metrics["errors"] += 1

    add_to_history("user", req.text)
    add_to_history("assistant", response)

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
        "response": response, "latency_ms": lat, "route": mode, "intent": intent,
        "spawn_hud": bool(hud_data),
        "hud_title": hud_data["title"] if hud_data else None,
        "hud_content": hud_data["content"] if hud_data else None,
        "hud_type": hud_data["type"] if hud_data else None,
    }

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

@app.post("/api/forge")
async def forge(req: ForgeRequest):
    system = """You are B.L.I.T.Z. FORGE mode — senior autonomous coding agent.
Fix/improve the code. Respond with: 1-2 sentence explanation + complete corrected code block.
Be precise. No verbose commentary."""
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

@app.post("/api/memory")
async def save_mem(req: MemoryRequest):
    data = load_memory()
    if req.key:
        data.setdefault("structured", {})[req.key] = req.entry
    data["entries"].append(req.entry)
    data["context"] = "\n".join(data["entries"][-30:])
    save_memory(data)
    return {"status": "saved", "total_entries": len(data["entries"])}

@app.get("/api/memory-view")
async def view_mem():
    data = load_memory()
    return {"content": "\n".join(data.get("entries", [])), "structured": data.get("structured", {})}

@app.post("/api/memory-clear")
async def clear_mem():
    save_memory({"entries": [], "context": "", "preferences": {}})
    _conversation_history.clear()
    return {"status": "cleared"}

@app.get("/api/history")
async def get_history():
    return {"history": _conversation_history}

@app.post("/api/history-clear")
async def clear_history():
    _conversation_history.clear()
    return {"status": "cleared"}

@app.get("/api/search")
async def web_search(q: str):
    if not tavily:
        return {"error": "Tavily not configured"}
    try:
        results = tavily.search(q, max_results=5, search_depth="advanced")
        return {"results": results.get("results", [])}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/scrape")
async def scrape(req: ScrapeRequest):
    import httpx, re as _re
    if not req.url.startswith("http"):
        return JSONResponse(status_code=400, content={"error": "Invalid URL"})
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BLITZ/4.0)"}) as client:
            r = await client.get(req.url)
            html = r.text
        html = _re.sub(r"<(script|style|nav|footer|header)[^>]*>[\s\S]*?</\1>", "", html, flags=_re.I)
        title_m = _re.search(r"<title[^>]*>(.*?)</title>", html, _re.I | _re.S)
        title = title_m.group(1).strip() if title_m else req.url
        text = _re.sub(r"<[^>]+>", " ", html)
        text = _re.sub(r"\s{3,}", "\n\n", text).strip()
        return {"title": title, "content": text[:6000], "url": req.url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/auto")
async def autonomous_execute(req: AutoRequest):
    intent = detect_intent(req.text)
    action_result = await handle_autonomous_action(intent, req.text)
    system = build_system_prompt([], "", intent)
    action_ctx = f"\n\n[SYSTEM ACTION EXECUTED]: {json.dumps(action_result)}" if action_result else ""
    try:
        r = await groq_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": req.text + action_ctx}],
            max_tokens=500, temperature=0.6
        )
        response = r.choices[0].message.content
    except Exception:
        response = f"Action executed: {action_result}"
    return {"response": response, "intent": intent, "action": action_result}

# ── Serve HTML ─────────────────────────────────────────────────────────────────
import pathlib as _pl
_HTML_DIR = _pl.Path(__file__).parent
_HTML_FILES = ["blitz_hub.html", "forge.html", "canvas.html", "nexus.html", "studio.html", "index.html"]

for _fname in _HTML_FILES:
    _path = _HTML_DIR / _fname
    if _path.exists():
        @app.get(f"/{_fname}", include_in_schema=False)
        async def _serve(f=_path):
            return FileResponse(f, media_type="text/html")

_CLEAN_ROUTES = {"hub": "blitz_hub.html", "forge": "forge.html", "canvas": "canvas.html", "nexus": "nexus.html", "studio": "studio.html"}
for _route, _file in _CLEAN_ROUTES.items():
    _fpath = _HTML_DIR / _file
    if _fpath.exists():
        @app.get(f"/{_route}", include_in_schema=False)
        async def _serve_clean(f=_fpath):
            return FileResponse(f, media_type="text/html")

@app.get("/")
async def root():
    return FileResponse(_pl.Path(__file__).parent / "index.html", media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
