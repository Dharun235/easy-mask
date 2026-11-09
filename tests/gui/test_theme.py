"""The calm theme: palettes, stylesheet and application-wide setup."""

from PySide6.QtGui import QPalette

from easy_mask.gui import theme


def test_light_and_dark_palettes_differ_and_are_valid():
    for colors in (theme.LIGHT, theme.DARK):
        p = theme.palette(colors)
        assert p.color(QPalette.ColorRole.Window).name().upper() == colors.window
        assert p.color(QPalette.ColorRole.Text).name().upper() == colors.text
    assert theme.LIGHT.window != theme.DARK.window


def test_stylesheet_uses_the_palette_colours():
    css = theme.stylesheet(theme.LIGHT)
    assert theme.LIGHT.accent in css and theme.LIGHT.accent_soft in css
    assert "QToolButton:checked" in css and "{" in css


def test_apply_theme_sets_fusion_palette_and_stylesheet(qapp, monkeypatch):
    old_style = qapp.style().objectName() or "Fusion"
    old_palette, old_sheet = qapp.palette(), qapp.styleSheet()
    monkeypatch.setattr(theme, "system_is_dark", lambda: True)
    try:
        colors = theme.apply_theme(qapp)
        assert colors is theme.DARK and theme.current() is theme.DARK
        # (the stylesheet wraps the style in a proxy whose name Qt reports as empty)
        assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == theme.DARK.window
        assert theme.DARK.accent in qapp.styleSheet()
    finally:
        qapp.setStyle(old_style)
        qapp.setPalette(old_palette)
        qapp.setStyleSheet(old_sheet)


def test_system_is_dark_returns_a_bool():
    assert isinstance(theme.system_is_dark(), bool)
