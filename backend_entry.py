"""Entry point for isolated app-local CPython (whose _pth ignores script paths)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from component_paths import configure
configure()

if __name__ == '__main__':
    from backend_process import contain_children
    contain_children()
    from server import main
    main()
