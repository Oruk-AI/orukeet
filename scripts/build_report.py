"""Compatibility entry point for the four-page LaTeX technical report."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_neurips_report.py'))], check=True)
