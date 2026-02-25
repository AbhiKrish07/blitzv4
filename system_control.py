"""
B.L.I.T.Z. System Control Router
─────────────────────────────────
Real PC control endpoints for the AI Cursor (Voice DOM).
Handles: app launching, brightness, volume, processes, shell commands,
         Spotify OAuth + Now-Playing proxy.

Mount in main.py:
    from system_control import router as sys_router
    app.include_router(sys_router)
"""

import os, subprocess, asyncio, json, time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/sys", tags=["system"])

# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def _try(fn, fallback=None):
    try:
        return fn()
    except Exception as e:
        print(f"[SYS WARN] {e}")
        return fallback

# ──────────────────────────────────────────────────────────────────
# APP LAUNCHER
# Windows: uses the START command / known app aliases
# ──────────────────────────────────────────────────────────────────

KNOWN_APPS = {
    # common names → shell command
    "notepad":        "notepad.exe",
    "calculator":     "calc.exe",
    "paint":          "mspaint.exe",
    "word":           "winword.exe",
    "excel":          "excel.exe",
    "powerpoint":     "POWERPNT.exe",
    "chrome":         "chrome.exe",
    "edge":           "msedge.exe",
    "firefox":        "firefox.exe",
    "explorer":       "explorer.exe",
    "file explorer":  "explorer.exe",
    "cmd":            "cmd.exe",
    "terminal":       "wt.exe",          # Windows Terminal
    "powershell":     "powershell.exe",
    "spotify":        "spotify.exe",
    "discord":        "discord.exe",
    "slack":          "slack.exe",
    "vscode":         "code.exe",
    "vs code":        "code.exe",
    "visual studio code": "code.exe",
    "task manager":   "taskmgr.exe",
    "control panel":  "control.exe",
    "settings":       "ms-settings:",    # URI scheme
    "display settings":          "ms-settings:display",
    "sound settings":            "ms-settings:sound",
    "network settings":          "ms-settings:network-wifi",
    "bluetooth settings":        "ms-settings:bluetooth",
    "apps settings":             "ms-settings:appsfeatures",
    "power settings":            "ms-settings:powersleep",
    "personalization":           "ms-settings:personalization",
    "wallpaper":                 "ms-settings:personalization-background",
    "lock screen":               "ms-settings:personalization-lockscreen",
    "snipping tool":             "snippingtool.exe",
    "snip":                      "snippingtool.exe",
    "clock":                     "ms-clock:",
    "maps":                      "bingmaps:",
    "camera":                    "microsoft.windows.camera:",
    "photos":                    "ms-photos:",
    "mail":                      "outlookmail:",
    "calendar":                  "outlookcal:",
    "store":                     "ms-windows-store:",
    "xbox":                      "xbox:",
    "Teams":                     "msteams.exe",
}

class AppRequest(BaseModel):
    name: str

@router.post("/open-app")
async def open_app(req: AppRequest):
    name = req.name.lower().strip()
    cmd = KNOWN_APPS.get(name)

    if not cmd:
        # Fuzzy match
        for k, v in KNOWN_APPS.items():
            if name in k or k in name:
                cmd = v
                break

    if not cmd:
        # Last resort: try to run it directly (user might know the exe)
        cmd = req.name

    try:
        if cmd.startswith("ms-") or cmd.endswith(":"):
            # URI scheme — use PowerShell to launch
            subprocess.Popen(
                ["powershell", "-Command", f"Start-Process '{cmd}'"],
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW
            )
        elif cmd.endswith(".exe") or "/" in cmd or "\\" in cmd:
            subprocess.Popen(
                cmd, shell=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                ["powershell", "-Command", f"Start-Process '{cmd}'"],
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW
            )
        return {"status": "launched", "app": name, "cmd": cmd}
    except Exception as e:
        raise HTTPException(400, f"Failed to open '{name}': {e}")

# ──────────────────────────────────────────────────────────────────
# SHELL / RUN COMMAND (restricted)
# ──────────────────────────────────────────────────────────────────
ALLOWED_SHELLS = {"explorer", "notepad", "calc", "mspaint", "taskmgr",
                  "snippingtool", "mmc", "wt", "cmd", "powershell", "code"}

class ShellRequest(BaseModel):
    command: str

@router.post("/shell")
async def shell_run(req: ShellRequest):
    """Run a whitelisted command and return stdout."""
    first_word = req.command.strip().split()[0].lower().replace(".exe", "")
    if first_word not in ALLOWED_SHELLS:
        raise HTTPException(403, f"Command '{first_word}' not in allowed list.")
    try:
        result = subprocess.run(
            req.command, shell=True, capture_output=True, text=True, timeout=10
        )
        return {"stdout": result.stdout[:2000], "stderr": result.stderr[:500], "rc": result.returncode}
    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Command timed out")
    except Exception as e:
        raise HTTPException(500, str(e))

# ──────────────────────────────────────────────────────────────────
# SCREEN BRIGHTNESS
# ──────────────────────────────────────────────────────────────────
@router.get("/brightness")
async def get_brightness():
    try:
        import screen_brightness_control as sbc
        val = sbc.get_brightness(display=0)
        return {"brightness": val[0] if isinstance(val, list) else val}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e), "fallback": True})

class BrightnessRequest(BaseModel):
    level: int   # 0–100

@router.post("/brightness")
async def set_brightness(req: BrightnessRequest):
    level = max(1, min(100, req.level))
    try:
        import screen_brightness_control as sbc
        sbc.set_brightness(level, display=0)
        return {"status": "ok", "brightness": level}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e), "fallback": True})

# ──────────────────────────────────────────────────────────────────
# SYSTEM VOLUME
# ──────────────────────────────────────────────────────────────────
def _get_volume_interface():
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

@router.get("/volume")
async def get_volume():
    try:
        vol = _get_volume_interface()
        pct = round(vol.GetMasterVolumeLevelScalar() * 100)
        muted = bool(vol.GetMute())
        return {"volume": pct, "muted": muted}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e), "fallback": True})

class VolumeRequest(BaseModel):
    level: int = -1   # 0–100, or -1 to skip
    mute: bool = None

@router.post("/volume")
async def set_volume(req: VolumeRequest):
    try:
        vol = _get_volume_interface()
        if req.level >= 0:
            vol.SetMasterVolumeLevelScalar(max(0.0, min(1.0, req.level / 100)), None)
        if req.mute is not None:
            vol.SetMute(int(req.mute), None)
        pct = round(vol.GetMasterVolumeLevelScalar() * 100)
        return {"status": "ok", "volume": pct, "muted": bool(vol.GetMute())}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e), "fallback": True})

# ──────────────────────────────────────────────────────────────────
# PROCESS LIST / KILL
# ──────────────────────────────────────────────────────────────────
@router.get("/processes")
async def list_processes():
    try:
        import psutil
        procs = [
            {"pid": p.pid, "name": p.info["name"], "cpu": p.info["cpu_percent"],
             "mem_mb": round(p.info["memory_info"].rss / 1024 / 1024, 1) if p.info["memory_info"] else 0}
            for p in psutil.process_iter(["name", "cpu_percent", "memory_info"])
            if p.info["name"]
        ]
        procs.sort(key=lambda x: x["mem_mb"], reverse=True)
        return {"processes": procs[:50]}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

class KillRequest(BaseModel):
    pid: int = -1
    name: str = ""

@router.post("/kill")
async def kill_process(req: KillRequest):
    try:
        import psutil
        killed = []
        for p in psutil.process_iter(["name", "pid"]):
            if (req.pid > 0 and p.pid == req.pid) or \
               (req.name and req.name.lower() in p.info["name"].lower()):
                p.kill()
                killed.append(p.info["name"])
        return {"killed": killed}
    except Exception as e:
        raise HTTPException(500, str(e))

# ──────────────────────────────────────────────────────────────────
# SYSTEM INFO
# ──────────────────────────────────────────────────────────────────
@router.get("/info")
async def system_info():
    try:
        import psutil, platform
        return {
            "os": platform.system(),
            "version": platform.version(),
            "cpu_pct": psutil.cpu_percent(interval=0.3),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "ram_used_pct": psutil.virtual_memory().percent,
            "disk_free_gb": round(psutil.disk_usage("/").free / 1e9, 1),
            "battery": psutil.sensors_battery()._asdict() if psutil.sensors_battery() else None,
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────────────────────────
# SCREENSHOT (for Siri-like context)
# ──────────────────────────────────────────────────────────────────
@router.get("/screenshot")
async def take_screenshot():
    """Take a screen capture and return as base64 JPEG."""
    try:
        import pyautogui, io, base64
        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"image_b64": b64, "width": img.width, "height": img.height}
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

# ──────────────────────────────────────────────────────────────────
# MOUSE / KEYBOARD (pyautogui)
# ──────────────────────────────────────────────────────────────────
class MouseRequest(BaseModel):
    x: int
    y: int
    action: str = "click"   # click | move | rightclick | doubleclick

@router.post("/mouse")
async def mouse_action(req: MouseRequest):
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        if req.action == "move":
            pyautogui.moveTo(req.x, req.y, duration=0.25)
        elif req.action == "click":
            pyautogui.click(req.x, req.y)
        elif req.action == "rightclick":
            pyautogui.rightClick(req.x, req.y)
        elif req.action == "doubleclick":
            pyautogui.doubleClick(req.x, req.y)
        return {"status": "ok", "action": req.action, "x": req.x, "y": req.y}
    except Exception as e:
        raise HTTPException(500, str(e))

class TypeRequest(BaseModel):
    text: str
    interval: float = 0.03

@router.post("/type")
async def type_text(req: TypeRequest):
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.typewrite(req.text, interval=req.interval)
        return {"status": "ok", "typed": req.text[:50]}
    except Exception as e:
        raise HTTPException(500, str(e))

class HotkeyRequest(BaseModel):
    keys: list[str]   # e.g. ["ctrl", "c"]

@router.post("/hotkey")
async def hotkey(req: HotkeyRequest):
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.hotkey(*req.keys)
        return {"status": "ok", "keys": req.keys}
    except Exception as e:
        raise HTTPException(500, str(e))

# ──────────────────────────────────────────────────────────────────
# SPOTIFY OAUTH  — PKCE flow (no client secret needed)
# Requires .env: SPOTIFY_CLIENT_ID, SPOTIFY_REDIRECT_URI
# ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_SCOPES    = "user-read-playback-state user-read-currently-playing user-modify-playback-state"

def _get_redirect_uri() -> str:
    """
    Resolve redirect URI at request time so Render's injected env vars
    are always picked up. Priority:
      1. SPOTIFY_REDIRECT_URI env var (explicit override)
      2. RENDER_EXTERNAL_URL (Render sets this automatically)
      3. localhost fallback for local dev
    """
    explicit = os.getenv("SPOTIFY_REDIRECT_URI", "").strip()
    if explicit:
        # Accept both bare URL and full callback URL
        if explicit.endswith("/callback"):
            return explicit
        return explicit.rstrip("/") + "/api/sys/spotify/callback"

    render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if render_url:
        return render_url.rstrip("/") + "/api/sys/spotify/callback"

    return "http://localhost:8000/api/sys/spotify/callback"

# In-memory token store (swap for DB in production)
_spotify_tokens: dict = {}

@router.get("/spotify/debug")
async def spotify_debug():
    """Check exactly what redirect URI and client ID the server is using.
    Visit https://blitz-v4.onrender.com/api/sys/spotify/debug in your browser."""
    return {
        "client_id_set": bool(SPOTIFY_CLIENT_ID),
        "client_id_preview": (SPOTIFY_CLIENT_ID[:8] + "...") if SPOTIFY_CLIENT_ID else "NOT SET",
        "redirect_uri_in_use": _get_redirect_uri(),
        "SPOTIFY_REDIRECT_URI_env": os.getenv("SPOTIFY_REDIRECT_URI", "NOT SET"),
        "RENDER_EXTERNAL_URL_env": os.getenv("RENDER_EXTERNAL_URL", "NOT SET"),
    }

@router.get("/spotify/auth")
async def spotify_auth():
    """Redirect user to Spotify auth page."""
    if not SPOTIFY_CLIENT_ID:
        return JSONResponse(status_code=503, content={
            "error": "SPOTIFY_CLIENT_ID not set in .env",
            "help": "Add SPOTIFY_CLIENT_ID and SPOTIFY_REDIRECT_URI to your .env file, then restart the server."
        })
    import urllib.parse, secrets, hashlib, base64
    verifier = secrets.token_urlsafe(64)
    _spotify_tokens["verifier"] = verifier
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    params = {
        "response_type": "code",
        "client_id": SPOTIFY_CLIENT_ID,
        "scope": SPOTIFY_SCOPES,
        "redirect_uri": _get_redirect_uri(),
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": "blitz",
    }
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)
    return RedirectResponse(url)

@router.get("/spotify/callback")
async def spotify_callback(code: str = "", error: str = "", state: str = ""):
    """Handle OAuth callback from Spotify."""
    if error:
        return JSONResponse(status_code=400, content={"error": error})
    if not code:
        return JSONResponse(status_code=400, content={"error": "No code received"})
    import httpx
    verifier = _spotify_tokens.get("verifier", "")
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://accounts.spotify.com/api/token", data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _get_redirect_uri(),
            "client_id": SPOTIFY_CLIENT_ID,
            "code_verifier": verifier,
        })
    if resp.status_code != 200:
        return JSONResponse(status_code=400, content={"error": resp.text})
    token_data = resp.json()
    _spotify_tokens["access_token"]  = token_data.get("access_token")
    _spotify_tokens["refresh_token"] = token_data.get("refresh_token")
    _spotify_tokens["expires_at"]    = time.time() + token_data.get("expires_in", 3600)
    # Return a page that sends token to the frontend and closes the popup
    html = """<!DOCTYPE html><html><head><title>Spotify Connected</title>
    <style>body{background:#0a0a14;color:#fff;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
    .box{text-align:center;padding:40px;border:1px solid rgba(167,139,250,.3);border-radius:16px;background:rgba(167,139,250,.07)}
    .check{font-size:48px;margin-bottom:16px}.title{font-size:20px;font-weight:700;color:#a78bfa;margin-bottom:8px}
    .sub{font-size:12px;color:rgba(255,255,255,.5)}</style></head><body>
    <div class="box"><div class="check">✅</div>
    <div class="title">Spotify Connected!</div>
    <div class="sub">This window will close automatically...</div></div>
    <script>
      if(window.opener){ window.opener.postMessage({type:'spotify_auth_ok'},'*'); }
      setTimeout(()=>window.close(), 1500);
    </script></body></html>"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(html)

async def _refresh_spotify_token():
    refresh = _spotify_tokens.get("refresh_token")
    if not refresh:
        return False
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://accounts.spotify.com/api/token", data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": SPOTIFY_CLIENT_ID,
        })
    if resp.status_code == 200:
        data = resp.json()
        _spotify_tokens["access_token"] = data.get("access_token")
        _spotify_tokens["expires_at"]   = time.time() + data.get("expires_in", 3600)
        if data.get("refresh_token"):
            _spotify_tokens["refresh_token"] = data["refresh_token"]
        return True
    return False

async def _spotify_headers():
    if not _spotify_tokens.get("access_token"):
        return None
    if time.time() > _spotify_tokens.get("expires_at", 0) - 60:
        await _refresh_spotify_token()
    tok = _spotify_tokens.get("access_token")
    return {"Authorization": f"Bearer {tok}"} if tok else None

@router.get("/spotify/status")
async def spotify_status():
    """Is Spotify connected?"""
    connected = bool(_spotify_tokens.get("access_token"))
    return {"connected": connected, "client_configured": bool(SPOTIFY_CLIENT_ID)}

@router.get("/spotify/now-playing")
async def spotify_now_playing():
    """Proxy the Spotify current-playback endpoint."""
    headers = await _spotify_headers()
    if not headers:
        return JSONResponse(status_code=401, content={"error": "Not authenticated", "auth_url": "/api/sys/spotify/auth"})
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.spotify.com/v1/me/player/currently-playing",
                                headers=headers)
    if resp.status_code == 204:
        return {"playing": False, "item": None}
    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
    data = resp.json()
    item = data.get("item", {})
    return {
        "playing":   data.get("is_playing", False),
        "progress_ms": data.get("progress_ms", 0),
        "item": {
            "id":       item.get("id"),
            "title":    item.get("name",""),
            "artist":   ", ".join(a["name"] for a in item.get("artists",[])),
            "album":    item.get("album",{}).get("name",""),
            "art_url":  (item.get("album",{}).get("images") or [{}])[0].get("url",""),
            "duration_ms": item.get("duration_ms", 0),
            "external_url": item.get("external_urls",{}).get("spotify",""),
        }
    }

@router.get("/spotify/devices")
async def spotify_devices():
    headers = await _spotify_headers()
    if not headers:
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.spotify.com/v1/me/player/devices", headers=headers)
    return resp.json()

class SpotifyControl(BaseModel):
    action: str       # play | pause | next | previous | seek | volume
    value: int = 0    # seek position_ms or volume 0-100
    device_id: str = ""

@router.post("/spotify/control")
async def spotify_control(req: SpotifyControl):
    headers = await _spotify_headers()
    if not headers:
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})
    import httpx
    base = "https://api.spotify.com/v1/me/player"
    params = {"device_id": req.device_id} if req.device_id else {}
    async with httpx.AsyncClient() as client:
        if req.action == "play":
            r = await client.put(f"{base}/play", headers=headers, params=params)
        elif req.action == "pause":
            r = await client.put(f"{base}/pause", headers=headers, params=params)
        elif req.action == "next":
            r = await client.post(f"{base}/next", headers=headers, params=params)
        elif req.action == "previous":
            r = await client.post(f"{base}/previous", headers=headers, params=params)
        elif req.action == "seek":
            r = await client.put(f"{base}/seek", headers=headers, params={**params, "position_ms": req.value})
        elif req.action == "volume":
            r = await client.put(f"{base}/volume", headers=headers, params={**params, "volume_percent": req.value})
        else:
            raise HTTPException(400, f"Unknown action: {req.action}")
    return {"status": "ok", "action": req.action, "http": r.status_code}

@router.put("/spotify/shuffle")
async def spotify_shuffle(state: bool = False):
    """Toggle shuffle on/off."""
    headers = await _spotify_headers()
    if not headers:
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.put(
            f"https://api.spotify.com/v1/me/player/shuffle?state={str(state).lower()}",
            headers=headers
        )
    return {"status": "ok", "shuffle": state, "http": r.status_code}

@router.put("/spotify/repeat")
async def spotify_repeat(state: str = "off"):
    """Set repeat mode: off | track | context."""
    headers = await _spotify_headers()
    if not headers:
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.put(
            f"https://api.spotify.com/v1/me/player/repeat?state={state}",
            headers=headers
        )
    return {"status": "ok", "repeat": state, "http": r.status_code}
