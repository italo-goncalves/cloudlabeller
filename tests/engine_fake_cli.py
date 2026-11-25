"""A stand-in Job process for the Engine tests: speaks the ``*_cli`` protocol.

``python -m tests.engine_fake_cli <exit_code> [sleep_s] [stop_file]``
"""

import json
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    code = int(argv[1])
    sleep_s = float(argv[2]) if len(argv) > 2 else 0.0
    stop = Path(argv[3]) if len(argv) > 3 else None
    print("PROGRESS 0.500 halfway", flush=True)
    print("a plain log line", flush=True)
    print("warning on stderr", file=sys.stderr, flush=True)
    deadline = time.monotonic() + sleep_s
    while time.monotonic() < deadline:
        if stop is not None and stop.exists():
            break
        time.sleep(0.05)
    if code == 0:
        print("RESULT " + json.dumps({"answer": 42}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
