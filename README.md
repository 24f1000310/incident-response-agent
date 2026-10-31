# GA5 Incident Response Agent

FastAPI + Gemini + PostgreSQL implementation for `ga5-incident-agent/v2`.

## Local run (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY and DATABASE_URL
$env:GEMINI_API_KEY="YOUR_GOOGLE_AI_STUDIO_KEY"
$env:DATABASE_URL="sqlite:///./incident_agent.db" # local testing only
uvicorn main:app --reload --port 8000
```

Health: `http://127.0.0.1:8000/health`

## Deploy on Render

1. Push this folder to a GitHub repository (never commit `.env`).
2. Create a persistent PostgreSQL database (Neon is fine) and copy its connection string.
3. Create a Render Web Service from the repository. Build: `pip install -r requirements.txt`; Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Set `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-2.5-flash`, and `DATABASE_URL` in Render environment variables. Use PostgreSQL in production; Render's local filesystem is not durable.
5. Submit only the base URL, e.g. `https://your-service.onrender.com` (no path, credentials, query, or fragment).

## Notes

- `sensitive` is never included in the model planner input or API state response. The full original request is persisted to support same-run conflict detection; if your grader requires strict at-rest minimization, replace request storage with a redacted request plus a keyed digest, while preserving replay checks.
- Diagnostic and effect tool calls are represented as dispatches for the grader to execute; this service does not call arbitrary tools itself.
- Test the assignment's exact receipt shapes against a staging deployment before using a limited Check/Save. This starter favors strict validation and safe suppression over guessing.
