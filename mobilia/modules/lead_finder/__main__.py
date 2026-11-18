"""Executa o Lead Finder isolado (desenvolvimento): ``python -m mobilia.modules.lead_finder``."""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication, QMainWindow

    from mobilia.ui import theme
    from mobilia.ui.widgets import StatusBar

    from .ui.page import LeadFinderPage

    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)
    win = QMainWindow()
    win.setWindowTitle("7xAi · Lead Finder")
    win.setStatusBar(StatusBar())
    page = LeadFinderPage(win)
    win.setCentralWidget(page)
    win.resize(1440, 900)
    win.show()
    page.on_show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
