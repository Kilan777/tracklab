"""TrackLab — video motion tracking & measurement (a friendlier Kinovea)."""

import os
import sys


def main() -> None:
    # Prefer native Wayland but fall back cleanly; OpenCV headless avoids Qt plugin clashes.
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication

    from .ui.main import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("TrackLab")
    apply_theme(app)
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.exists(args[0]):
        w.open_path(os.path.abspath(args[0]))
    sys.exit(app.exec())


def apply_theme(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in {
        QPalette.Window: "#181b21", QPalette.WindowText: "#d7dce6", QPalette.Base: "#12151a",
        QPalette.AlternateBase: "#1c2027", QPalette.Text: "#d7dce6", QPalette.Button: "#232831",
        QPalette.ButtonText: "#d7dce6", QPalette.Highlight: "#3987e5", QPalette.HighlightedText: "#ffffff",
        QPalette.ToolTipBase: "#232831", QPalette.ToolTipText: "#d7dce6", QPalette.PlaceholderText: "#6b7486",
        QPalette.Link: "#4da3ff",
    }.items():
        pal.setColor(role, QColor(c))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#5a6272"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5a6272"))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#5a6272"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


STYLE = """
QGroupBox { border: 1px solid #2a303b; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8fa0bb; font-weight: bold; }
QPushButton { background: #262c36; border: 1px solid #343c4a; border-radius: 5px; padding: 4px 8px; }
QPushButton:hover { background: #2e3542; border-color: #4a5568; }
QPushButton:pressed { background: #1d222a; }
QToolBar { background: #1b1f26; border-bottom: 1px solid #2a303b; spacing: 3px; padding: 3px; }
QToolBar QToolButton { padding: 5px 9px; border-radius: 5px; color: #d7dce6; }
QToolBar QToolButton:hover { background: #2a313c; }
QToolBar QToolButton:checked { background: #3987e5; color: white; }
QToolBar QToolButton:disabled { color: #4b5363; }
QListWidget { border: 1px solid #2a303b; border-radius: 4px; }
QListWidget::item { padding: 2px; }
QListWidget::item:selected { background: #2b3a52; color: #ffffff; }
QLabel#hint { background: #1f2a3a; color: #b9c9e4; padding: 6px 10px; border-bottom: 1px solid #2a303b; }
QLabel#hint[alert="true"] { background: #5a1e24; color: #ffd9dc; font-weight: bold; }
QWidget#transport { background: #181b21; border-top: 1px solid #2a303b; }
QWidget#transport QLabel { color: #c3cbd8; }
QSplitter::handle { background: #232831; }
QStatusBar { color: #9aa5b8; }
QWidget#welcome { background: #0d0f13; }
QLabel#wtitle { font-size: 34px; font-weight: 800; color: #e8edf5; }
QLabel#wsub { color: #8a94a7; font-size: 14px; }
QLabel#whead { color: #8fa0bb; font-weight: bold; margin-top: 8px; }
QPushButton#excard { background: #161a21; border: 1px solid #2a303b; border-radius: 10px; text-align: left; padding: 0; }
QPushButton#excard:hover { background: #1c2230; border-color: #3987e5; }
QLabel#wcard { background: #161a21; border: 1px solid #2a303b; border-radius: 8px; padding: 12px; min-height: 140px; }
QPushButton#primary { background: #3987e5; border-color: #3987e5; color: white; font-weight: bold; padding: 6px 14px; }
QPushButton#primary:hover { background: #4a95ee; }
QPushButton#guidego { background: #3987e5; border-color: #3987e5; color: white; font-weight: bold; padding: 4px 2px; }
QPushButton#link { background: transparent; border: none; color: #4da3ff; text-align: left; padding: 3px 2px; font-size: 13px; }
QPushButton#link:hover { color: #7fbcff; text-decoration: underline; }

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #12151a; color: #d7dce6; border: 1px solid #343c4a; border-radius: 4px; padding: 2px 4px; selection-background-color: #3987e5; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled { color: #5a6272; }
QComboBox QAbstractItemView { background: #1c2027; color: #d7dce6; selection-background-color: #3987e5; }
QPushButton { color: #d7dce6; }
QPushButton:disabled { color: #5a6272; }
"""

if __name__ == "__main__":
    main()
