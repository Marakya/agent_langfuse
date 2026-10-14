"""Немедленный вывод результатов этапов в терминал CLI и Streamlit."""

import json
from datetime import datetime


def progress(stage: str, **result):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{timestamp} [{stage}] {json.dumps(result, ensure_ascii=False, default=str)}", flush=True)
