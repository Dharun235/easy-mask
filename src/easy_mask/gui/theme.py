"""A calm, minimal look: cool Nordic neutrals, one blue accent, flat panels, quiet chrome.

The aim is that the image and the work stand out and the tool fades into the background.
:func:`apply_theme` sets Qt's Fusion style (so it looks the same on every OS) with a light
or dark palette following the system setting, plus a small stylesheet.
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Colors:
    """One palette: everything else is derived from these few colours."""

    window: str  # app background
    surface: str  # panels, lists
    raised: str  # hover
    line: str  # hairlines, separators
    text: str
    muted: str  # secondary text, section titles
    canvas: str  # behind the image
    accent: str  # the single accent: active tool, selection
    accent_soft: str  # accent tint for backgrounds


LIGHT = Colors(
    window="#F5F6F8",
    surface="#FFFFFF",
    raised="#ECEEF2",
    line="#E1E4EA",
    text="#1D2129",
    muted="#7A818E",
    canvas="#E9ECF0",
    accent="#3B6FB6",
    accent_soft="#E3ECF8",
)
DARK = Colors(
    window="#17191D",
    surface="#1E2126",
    raised="#282C33",
    line="#30353D",
    text="#E8EAEE",
    muted="#8A919C",
    canvas="#111316",
    accent="#6E9BE0",
    accent_soft="#24324A",
)


def system_is_dark() -> bool:
    """True if the OS is in dark mode (Qt 6.5+ reports it; otherwise assume light)."""
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    return scheme is not None and scheme() == Qt.ColorScheme.Dark


def current() -> Colors:
    """The palette in use for this session."""
    return DARK if system_is_dark() else LIGHT


def palette(colors: Colors) -> QPalette:
    """A Qt palette built from ``colors``."""
    p = QPalette()
    role = QPalette.ColorRole
    for r, value in (
        (role.Window, colors.window),
        (role.WindowText, colors.text),
        (role.Base, colors.surface),
        (role.AlternateBase, colors.raised),
        (role.Text, colors.text),
        (role.Button, colors.window),
        (role.ButtonText, colors.text),
        (role.ToolTipBase, colors.text),
        (role.ToolTipText, colors.window),
        (role.PlaceholderText, colors.muted),
        (role.Highlight, colors.accent_soft),
        (role.HighlightedText, colors.text),
        (role.Mid, colors.line),
        (role.Midlight, colors.raised),
        (role.Light, colors.surface),
        (role.Dark, colors.line),
        (role.Link, colors.accent),
    ):
        p.setColor(r, QColor(value))
    p.setColor(QPalette.ColorGroup.Disabled, role.Text, QColor(colors.muted))
    p.setColor(QPalette.ColorGroup.Disabled, role.WindowText, QColor(colors.muted))
    p.setColor(QPalette.ColorGroup.Disabled, role.ButtonText, QColor(colors.muted))
    return p


ICONS = Path(__file__).parent / "icons"


def _mark(name: str, c: Colors) -> str:
    """Path of a tick drawn for this palette (``name.svg`` light, ``name-dark.svg`` dark)."""
    suffix = "-dark" if c is DARK else ""
    return (ICONS / f"{name}{suffix}.svg").as_posix()


def stylesheet(c: Colors) -> str:
    """Flat panels, rounded controls, quiet section titles."""
    check, done = _mark("check", c), _mark("done", c)  # ticks drawn for this palette
    return f"""
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator, QTreeView::indicator, QListView::indicator {{
    width: 15px; height: 15px; border: 1px solid {c.muted}; border-radius: 4px;
    background: {c.surface};
}}
QCheckBox::indicator:checked, QTreeView::indicator:checked, QListView::indicator:checked {{
    background: {c.accent}; border-color: {c.accent}; image: url({check});
}}
QTreeView::indicator:indeterminate {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {c.surface}, stop:0.4 {c.surface},
        stop:0.41 {c.accent}, stop:0.6 {c.accent}, stop:0.61 {c.surface}, stop:1 {c.surface});
    border-color: {c.accent};
}}
QListWidget#files::indicator {{ border: none; background: transparent; image: none; }}
QListWidget#files::indicator:checked {{ image: url({done}); }}
QCheckBox::indicator:disabled {{ border-color: {c.line}; background: {c.raised}; }}
QRadioButton {{ spacing: 8px; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {c.muted}; border-radius: 8px;
    background: {c.surface};
}}
QRadioButton::indicator:checked {{
    border-color: {c.accent};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {c.accent}, stop:0.45 {c.accent}, stop:0.55 {c.surface}, stop:1 {c.surface});
}}

QMainWindow, QDialog {{ background: {c.window}; }}
QToolBar {{
    background: {c.window}; border: none; border-bottom: 1px solid {c.line};
    padding: 6px 10px; spacing: 4px;
}}
QToolBar::separator {{ width: 1px; margin: 8px 8px; background: {c.line}; }}
QToolButton {{
    border: none; border-radius: 8px; padding: 6px; color: {c.text};
}}
QToolButton:hover {{ background: {c.raised}; }}
QToolButton:checked {{ background: {c.accent_soft}; }}
QToolButton:disabled {{ color: {c.muted}; }}
QToolButton[popupMode="1"] {{ padding-right: 16px; }}
QToolButton::menu-button {{ border: none; width: 14px; border-radius: 6px; }}
QToolButton::menu-button:hover {{ background: {c.raised}; }}
QDockWidget {{ color: {c.muted}; font-size: 11px; }}
QDockWidget::title {{
    padding: 10px 12px 4px 12px; background: {c.window}; text-transform: uppercase;
}}
QListWidget, QTableWidget, QTreeWidget {{
    background: {c.surface}; border: 1px solid {c.line}; border-radius: 10px;
    padding: 4px; outline: 0; color: {c.text}; font-size: 13px;
}}
QListWidget::item, QTreeWidget::item {{ padding: 4px 6px; border-radius: 6px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {c.raised}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {c.accent_soft}; color: {c.text};
}}
QHeaderView::section {{
    background: {c.surface}; color: {c.muted}; border: none; padding: 4px 6px;
}}
QStatusBar {{ background: {c.window}; color: {c.muted}; border-top: 1px solid {c.line}; }}
QStatusBar QLabel {{ color: {c.muted}; padding: 0 8px; }}
QStatusBar::item {{ border: none; }}
QPushButton {{
    background: {c.surface}; border: 1px solid {c.line}; border-radius: 9px;
    padding: 8px 16px; color: {c.text};
}}
QPushButton:hover {{ background: {c.raised}; }}
QPushButton:default {{ background: {c.accent}; color: {c.window}; border-color: {c.accent}; }}
QLineEdit, QSpinBox, QComboBox {{
    background: {c.surface}; border: 1px solid {c.line}; border-radius: 8px; padding: 5px 8px;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {c.accent}; }}
QProgressBar {{
    background: {c.raised}; border: none; border-radius: 3px; max-height: 6px;
}}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 3px; }}
QMenu {{ background: {c.surface}; border: 1px solid {c.line}; padding: 6px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c.raised}; color: {c.text}; }}
QToolTip {{
    background: {c.text}; color: {c.window}; border: none; padding: 5px 8px;
    border-radius: 6px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c.line}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 0; }}
"""


def apply_theme(app: QApplication) -> Colors:
    """Use the calm theme for the whole application; returns the colours in use.

    It follows the system: switching light / dark while easy-mask runs restyles it.
    Menus show no icons (like native Mac menus), so every entry lines up the same.
    """
    app.setStyle("Fusion")
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus)
    colors = _restyle(app)
    hints = QGuiApplication.styleHints()
    if hasattr(hints, "colorSchemeChanged"):
        hints.colorSchemeChanged.connect(lambda _scheme: _restyle(app))
    return colors


def _restyle(app: QApplication) -> Colors:
    colors = current()
    app.setPalette(palette(colors))
    app.setStyleSheet(stylesheet(colors))
    return colors
