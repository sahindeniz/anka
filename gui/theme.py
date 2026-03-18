"""
Astro Mastro Pro — Theme
Dark / Light PyQt6 stylesheet
"""

DARK = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-size: 10pt;
}
QMenuBar {
    background-color: #252525;
    color: #e0e0e0;
    border-bottom: 1px solid #333;
}
QMenuBar::item:selected { background-color: #3a3a3a; }
QMenu {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #444;
}
QMenu::item:selected { background-color: #3d6a9e; }
QToolBar {
    background-color: #252525;
    border: none;
    spacing: 4px;
    padding: 2px 4px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #cccccc;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
    min-width: 60px;
}
QToolBar QToolButton:hover {
    background-color: #3a3a3a;
    border-color: #555;
}
QToolBar QToolButton:pressed {
    background-color: #285299;
    border-color: #3d6a9e;
    color: #ffffff;
}
QToolBar QToolButton:checked {
    background-color: #285299;
    border-color: #3d6a9e;
    color: #ffffff;
}
QScrollArea, QScrollBar {
    background-color: #1e1e1e;
}
QScrollBar:vertical {
    width: 10px;
    background: #252525;
}
QScrollBar::handle:vertical {
    background: #555;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #777; }
QScrollBar:horizontal {
    height: 10px;
    background: #252525;
}
QScrollBar::handle:horizontal {
    background: #555;
    border-radius: 4px;
    min-width: 20px;
}
QGroupBox {
    border: 1px solid #444;
    margin-top: 6px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 12px;
    color: #88aaff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}
QLabel { color: #cccccc; }
QPushButton {
    background-color: #3a3a3a;
    color: #dddddd;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
}
QPushButton:hover { background-color: #4a4a4a; border-color: #777; }
QPushButton:pressed { background-color: #285299; }
QPushButton:disabled { color: #666; background-color: #2a2a2a; border-color: #444; }
QSlider::groove:horizontal {
    background: #444;
    height: 5px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #88aaff;
    width: 13px;
    height: 13px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #aaccff; }
QComboBox {
    background-color: #2e2e2e;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 3px 6px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    color: #e0e0e0;
    selection-background-color: #285299;
}
QSpinBox, QDoubleSpinBox {
    background-color: #2e2e2e;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 2px 4px;
}
QCheckBox { color: #cccccc; spacing: 5px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666;
    background: #333;
    border-radius: 3px;
}
QCheckBox::indicator:checked { background: #285299; border-color: #88aaff; }
QListWidget {
    background-color: #252525;
    color: #cccccc;
    border: 1px solid #444;
    border-radius: 3px;
}
QListWidget::item:selected { background-color: #285299; color: #fff; }
QListWidget::item:hover { background-color: #333; }
QProgressBar {
    background-color: #2a2a2a;
    border: 1px solid #555;
    border-radius: 3px;
    text-align: center;
    color: #ccc;
}
QProgressBar::chunk { background-color: #285299; }
QSplitter::handle { background: #333; }
QTabWidget::pane { border: 1px solid #444; }
QTabBar::tab {
    background: #2a2a2a;
    color: #aaa;
    padding: 5px 10px;
    border-radius: 3px 3px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #1e1e1e; color: #fff; border-bottom: 2px solid #88aaff; }
QTabBar::tab:hover { background: #333; color: #ccc; }
QStatusBar { background-color: #252525; color: #aaa; border-top: 1px solid #333; }
QDockWidget::title {
    background: #252525;
    padding: 4px;
    font-weight: bold;
    color: #88aaff;
}
QHeaderView::section {
    background-color: #2a2a2a;
    color: #ccc;
    border: 1px solid #444;
    padding: 3px;
}
"""

LIGHT = """
QMainWindow, QWidget {
    background-color: #f0f0f0;
    color: #1a1a1a;
    font-size: 10pt;
}
QMenuBar {
    background-color: #e8e8e8;
    color: #1a1a1a;
    border-bottom: 1px solid #ccc;
}
QMenuBar::item:selected { background-color: #d0d0d0; }
QMenu {
    background-color: #f5f5f5;
    color: #1a1a1a;
    border: 1px solid #bbb;
}
QMenu::item:selected { background-color: #3d6a9e; color: #fff; }
QToolBar {
    background-color: #e8e8e8;
    border: none;
    spacing: 4px;
    padding: 2px 4px;
}
QToolBar QToolButton {
    background-color: transparent;
    color: #333;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
    min-width: 60px;
}
QToolBar QToolButton:hover { background-color: #ddd; border-color: #bbb; }
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {
    background-color: #3d6a9e;
    color: #fff;
}
QGroupBox {
    border: 1px solid #bbb;
    margin-top: 6px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 12px;
    color: #285299;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QLabel { color: #333; }
QPushButton {
    background-color: #e0e0e0;
    color: #222;
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: bold;
}
QPushButton:hover { background-color: #d0d0d0; border-color: #999; }
QPushButton:pressed { background-color: #3d6a9e; color: #fff; }
QSlider::groove:horizontal { background: #ccc; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #3d6a9e;
    width: 13px; height: 13px;
    margin: -4px 0;
    border-radius: 6px;
}
QComboBox {
    background-color: #fff;
    color: #222;
    border: 1px solid #bbb;
    border-radius: 3px;
    padding: 3px 6px;
}
QCheckBox { color: #333; spacing: 5px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #999; background: #fff; border-radius: 3px; }
QCheckBox::indicator:checked { background: #3d6a9e; border-color: #285299; }
QStatusBar { background-color: #e8e8e8; color: #555; border-top: 1px solid #ccc; }
"""


def get_stylesheet(theme: str = "dark") -> str:
    return DARK if theme == "dark" else LIGHT
