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


def test_ticks_are_drawn_for_each_palette():
    assert "check-dark.svg" in theme.stylesheet(theme.DARK)
    assert "check.svg" in theme.stylesheet(theme.LIGHT)
    assert "check-dark.svg" not in theme.stylesheet(theme.LIGHT)


def test_switching_light_dark_while_running_retints_toolbar_icons(qapp, window):
    from PySide6.QtCore import QSize

    def icon_colour():
        image = window._export_action.icon().pixmap(QSize(48, 48)).toImage()
        from collections import Counter

        pixels = (image.pixelColor(x, y) for x in range(48) for y in range(48))
        return Counter(p.name() for p in pixels if p.alpha() == 255).most_common(1)[0][0]

    old_palette, old_sheet = qapp.palette(), qapp.styleSheet()
    try:
        for colors in (theme.LIGHT, theme.DARK):  # what the theme does on an OS switch
            qapp.setPalette(theme.palette(colors))
            qapp.setStyleSheet(theme.stylesheet(colors))
            assert icon_colour().upper() == colors.text
    finally:
        qapp.setPalette(old_palette)
        qapp.setStyleSheet(old_sheet)
