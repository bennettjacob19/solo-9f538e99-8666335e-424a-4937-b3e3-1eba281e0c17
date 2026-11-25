"""Test-only entrypoint: isolated interpreter (-I) with the project on sys.path.

Using -I keeps the server process immune to planted modules in the working
directory or site packages; the project path is injected explicitly here.
"""

import os
import sys

sys.path.insert(0, os.environ["APP_HOME"])

from app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
