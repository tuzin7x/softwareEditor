"""Ponto de entrada do executável (PyInstaller)."""
import sys

from mobilia.ui.app import main

if __name__ == "__main__":
    sys.exit(main())
