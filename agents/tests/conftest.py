"""Make ``agents`` importable however pytest is invoked.

Without this, running pytest from somewhere other than the repository root
fails with ImportError rather than a useful message.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
