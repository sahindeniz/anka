import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PyTorch MUST be imported BEFORE PyQt6 to avoid DLL conflicts on Windows.
# Some Windows installs raise OSError instead of ImportError when torch DLLs
# exist but fail to initialize, so treat that as "torch unavailable".
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass

from gui.app import start_app
if __name__ == "__main__":
    start_app()
