"""Start CaM-PDA. The program asks for paths; no source editing is needed."""
from pathlib import Path
import sys

# Also work when launched by absolute path, or from an IDE's Run button.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

if __name__ == '__main__':
    from cam_pda.cli import main
    raise SystemExit(main())
