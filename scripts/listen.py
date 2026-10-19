"""Start a private loopback-only listening/transcription interface."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import private_directory
from src.evaluation.listening import serve

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--transcript-first", action="store_true", help="Explicit pilot variant, logged per judgment")
    a = p.parse_args()
    serve(private_directory(a.run), a.port, a.transcript_first)
