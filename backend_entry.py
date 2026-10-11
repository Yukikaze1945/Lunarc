"""Entry point for isolated app-local CPython (whose _pth ignores script paths)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == '__main__':
    from server import main
    main()
