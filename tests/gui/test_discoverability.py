"""Welcome page, toolbar icons and tooltips, and the keyboard shortcut list."""

from easy_mask.gui.icons import ICON_DIR, icon
from easy_mask.gui.main_window import TOOLS, Settings
from easy_mask.gui.shortcuts import shortcut_rows


def test_every_tool_has_an_icon_and_a_tooltip_with_its_key(make_window):
    win = make_window(Settings())
    for text, mode, key, icon_name, _group in TOOLS:
        assert (ICON_DIR / f"{icon_name}.svg").is_file()
        action = win.tool_actions[mode]
        assert not action.icon().isNull()
        assert action.toolTip() == f"{text} ({key})"
    assert icon("does-not-exist").isNull()


def test_toolbar_sections_follow_tool_groups(make_window):
    from PySide6.QtWidgets import QToolBar

    win = make_window(Settings())
    toolbar = win.findChild(QToolBar, "Tools")
    separators = sum(1 for a in toolbar.actions() if a.isSeparator())
    assert separators == len({group for *_, group in TOOLS}) - 1


def test_welcome_page_until_an_image_is_open(make_window, image_dir):
    win = make_window(Settings())
    assert win.pages.currentWidget() is win.welcome
    win.open_path(image_dir)
    assert win.pages.currentWidget() is win.canvas


def test_welcome_recent_folders_open_on_click(make_window, image_dir):
    first = make_window(Settings())
    first.open_path(image_dir)
    second = make_window(Settings())
    item = second.welcome.recent_list.item(0)
    assert item.text() == str(image_dir.resolve())
    second.welcome.recent_list.itemClicked.emit(item)
    assert second.image_path is not None and second.image_path.parent == image_dir.resolve()


def test_welcome_without_recent_folders_shows_placeholder(make_window):
    win = make_window(Settings())
    placeholder = win.welcome.recent_list.item(0)
    assert placeholder.text() == "(nothing yet)"
    win.welcome.recent_list.itemClicked.emit(placeholder)  # does nothing
    assert win.image_path is None


def test_shortcut_list_is_built_from_the_menus(make_window):
    win = make_window(Settings())
    rows = shortcut_rows(win)
    commands = {command for _menu, _keys, command in rows}
    assert {"Polygon", "AI Segment", "Use Label 1", "Keyboard Shortcuts"} <= commands
    win.show_shortcuts()
    win.show_shortcuts()  # reuses the same window
    assert win.shortcuts_dialog.isVisible()
    assert win.shortcuts_dialog.table.rowCount() == len(rows)
