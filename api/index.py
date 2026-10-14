"""Vercel entrypoint of the public demo API: backend/app's FastAPI app as one Python function.

Vercel's FastAPI preset finds this file (api/index.py) and serves its module-level `app`, with
every request path unchanged (/api/health reaches the route /api/health). The project's settings
are in vercel.json at the repository root, the guide in docs/deploy.md.

The function installs only the root requirements.txt: no torch, Chroma or OpenAI client. So it is
always the public demo, whatever the project's variables say: PUBLIC_DEMO is set to true here,
before the settings are read, and ENVIRONMENT defaults to production, as in the Hugging Face Space
image. The app finds data/eval and data/index from backend/app's own location, so the bundle keeps
the repository's layout; nothing is written to disk.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))  # the app package is backend/app

os.environ["PUBLIC_DEMO"] = "true"
os.environ.setdefault("ENVIRONMENT", "production")

from app.main import create_app  # noqa: E402  (needs the path and environment above)

app = create_app()
