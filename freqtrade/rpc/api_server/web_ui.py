from html import escape
from pathlib import Path

from fastapi import APIRouter
from fastapi.exceptions import HTTPException
from starlette.responses import FileResponse, HTMLResponse


router_ui = APIRouter(include_in_schema=False, tags=["Web UI"])


@router_ui.get("/favicon.ico")
async def favicon():
    return FileResponse(str(Path(__file__).parent / "ui/favicon.ico"))


@router_ui.get("/fallback_file.html")
async def fallback():
    return FileResponse(str(Path(__file__).parent / "ui/fallback_file.html"))


@router_ui.get("/ui_version")
async def ui_version():
    from freqtrade.commands.deploy_ui import read_ui_version

    uibase = Path(__file__).parent / "ui/installed/"
    version = read_ui_version(uibase)

    return {
        "version": version if version else "not_installed",
    }


@router_ui.get("/login-info", response_class=HTMLResponse)
async def login_info():
    """
    Labs helper: show API/FreqUI credentials for quick copy-paste.
    """
    from freqtrade.rpc.api_server.webserver import ApiServer

    api_cfg = (ApiServer._config or {}).get("api_server", {})
    username = str(api_cfg.get("username") or "")
    password = str(api_cfg.get("password") or "")
    listen_port = api_cfg.get("listen_port") or 8080
    ui_url = f"http://localhost:{listen_port}"
    bot_name = (ApiServer._config or {}).get("bot_name") or "freqtrade"

    username_html = escape(username)
    password_html = escape(password)
    ui_url_html = escape(ui_url)
    bot_name_html = escape(str(bot_name))

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Freqtrade — infos de connexion</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --accent: #3d9cf0;
      --border: #2a3648;
      --ok: #3ecf8e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, #1d3a5c 0%, transparent 45%),
        var(--bg);
      color: var(--text);
      padding: 1.5rem;
    }}
    .card {{
      width: min(520px, 100%);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
    }}
    h1 {{
      margin: 0 0 0.35rem;
      font-size: 1.35rem;
      font-weight: 650;
    }}
    .sub {{
      margin: 0 0 1.25rem;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .row {{
      display: grid;
      grid-template-columns: 110px 1fr auto;
      gap: 0.65rem;
      align-items: center;
      margin-bottom: 0.75rem;
    }}
    .label {{
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .value {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      background: #0c1118;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.65rem 0.75rem;
      overflow-x: auto;
      white-space: nowrap;
      user-select: all;
    }}
    button {{
      appearance: none;
      border: 1px solid var(--border);
      background: #243044;
      color: var(--text);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      cursor: pointer;
      font: inherit;
      font-size: 0.85rem;
      white-space: nowrap;
    }}
    button:hover {{ border-color: var(--accent); }}
    button.copied {{
      background: rgba(62, 207, 142, 0.15);
      border-color: var(--ok);
      color: var(--ok);
    }}
    .actions {{
      display: flex;
      gap: 0.75rem;
      margin-top: 1.25rem;
      flex-wrap: wrap;
    }}
    .actions a, .actions button.primary {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      background: var(--accent);
      color: #041018;
      border: none;
      border-radius: 999px;
      padding: 0.75rem 1.1rem;
      font-weight: 650;
    }}
    .hint {{
      margin-top: 1rem;
      color: var(--muted);
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
  <main class="card">
    <h1>Connexion FreqUI</h1>
    <p class="sub">Bot <strong>{bot_name_html}</strong> — clique pour copier, puis colle dans le login.</p>

    <div class="row">
      <div class="label">URL</div>
      <div class="value" id="url">{ui_url_html}</div>
      <button type="button" data-copy="url">Copier</button>
    </div>
    <div class="row">
      <div class="label">Username</div>
      <div class="value" id="username">{username_html}</div>
      <button type="button" data-copy="username">Copier</button>
    </div>
    <div class="row">
      <div class="label">Password</div>
      <div class="value" id="password">{password_html}</div>
      <button type="button" data-copy="password">Copier</button>
    </div>

    <div class="actions">
      <button type="button" class="primary" id="copy-all">Tout copier</button>
      <a href="{ui_url_html}/">Ouvrir FreqUI</a>
    </div>
    <p class="hint">Page Labs locale — ne pas exposer sur Internet.</p>
  </main>
  <script>
    async function copyText(text, button) {{
      await navigator.clipboard.writeText(text);
      if (!button) return;
      const previous = button.textContent;
      button.textContent = "Copié";
      button.classList.add("copied");
      setTimeout(() => {{
        button.textContent = previous;
        button.classList.remove("copied");
      }}, 1200);
    }}

    document.querySelectorAll("[data-copy]").forEach((button) => {{
      button.addEventListener("click", () => {{
        const id = button.getAttribute("data-copy");
        const el = document.getElementById(id);
        copyText(el.textContent, button);
      }});
    }});

    document.getElementById("copy-all").addEventListener("click", (event) => {{
      const payload = [
        "URL: " + document.getElementById("url").textContent,
        "Username: " + document.getElementById("username").textContent,
        "Password: " + document.getElementById("password").textContent,
      ].join("\\n");
      copyText(payload, event.currentTarget);
    }});
  </script>
</body>
</html>
"""


@router_ui.get("/{rest_of_path:path}")
async def index_html(rest_of_path: str):
    """
    Emulate path fallback to index.html.
    """
    if rest_of_path.startswith(("api", ".")):
        raise HTTPException(status_code=404, detail="Not Found")
    uibase = (Path(__file__).parent / "ui/installed/").resolve()
    filename = (uibase / rest_of_path).resolve()
    # It's security relevant to check "relative_to".
    # Without this, Directory-traversal is possible.
    media_type: str | None = None
    if filename.suffix == ".js":
        # Force text/javascript for .js files - Circumvent faulty system configuration
        media_type = "application/javascript"
    if filename.is_file() and filename.is_relative_to(uibase):
        return FileResponse(str(filename), media_type=media_type)

    index_file = uibase / "index.html"
    if not index_file.is_file():
        return FileResponse(str(uibase.parent / "fallback_file.html"))
    # Fall back to index.html, as indicated by vue router docs
    return FileResponse(str(index_file))
