import sys
import os
from pathlib import Path

os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

from streamlit.web.cli import main


bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
sys.path.insert(0, str(bundle_dir))

app_path = bundle_dir / "app.py"
sys.argv = [
    "streamlit",
    "run",
    str(app_path),
    "--server.address=127.0.0.1",
    "--server.headless=false",
    "--browser.gatherUsageStats=false",
]

raise SystemExit(main())