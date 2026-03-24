import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PyTorch MUST be imported BEFORE PyQt6 to avoid DLL conflicts on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

from gui.app import start_app
if __name__ == "__main__":
    start_app()
