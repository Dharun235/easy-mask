"""Welcome page, toolbar icons and tooltips, and the keyboard shortcut list."""

from PySide6.QtCore import Qt

from easy_mask.gui.icons import ICON_DIR, icon
from easy_mask.gui.main_window import KEYS_ONLY, TOOLS, Settings
from easy_mask.gui.shortcuts import shortcut_rows


def test_every_tool_has_an_icon_a_word_and_a_tooltip(make_window):
    win = make_window(Settings())
    for text, short, mode, _key, icon_name, _group in TOOLS:
        assert (ICON_DIR / f"{icon_name}.svg").is_file()
        action = win.tool_actions[mode]
        assert not action.icon().isNull()
        assert action.toolTip() == text
        assert action.iconText() == short
    assert icon("does-not-exist").isNull()


def test_toolbar_is_one_row_of_labelled_buttons(make_window):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolBar

    win = make_window(Settings())
    (toolbar,) = win.findChildren(QToolBar)
    assert toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextUnderIcon
    assert all(a.iconText() for a in toolbar.actions() if not a.isSeparator() and a.text())
    actions = [a for a in toolbar.actions() if not a.isSeparator()]
    texts = [a.text() for a in actions]
    assert texts.index("Open Folder...") < texts.index("Select / Edit")  # file first
    assert texts[-1] == "Export Again"  # export alone on the right
    separators = sum(1 for a in toolbar.actions() if a.isSeparator())
    groups = {group for *_, group in TOOLS} - {KEYS_ONLY}
    assert separators == 3 + len(groups) - 1  # file groups + tool groups
    for kept_out in ("Line", "Point", "Circle", "Magic wand"):
        assert kept_out not in texts  # keys only: the toolbar stays short


def test_working_tools_are_disabled_until_an_image_is_open(make_window, image_dir):
    win = make_window(Settings())
    assert not win.tool_actions["ai"].isEnabled()
    assert not win.propagate_button_action.isEnabled()
    win.open_path(image_dir)
    assert win.tool_actions["ai"].isEnabled() and win.propagate_button_action.isEnabled()


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
    assert {"Polygon", "Magic wand", "Use Label 1", "Keyboard Shortcuts"} <= commands
    win.show_shortcuts()
    win.show_shortcuts()  # reuses the same window
    assert win.shortcuts_dialog.isVisible()
    assert win.shortcuts_dialog.table.rowCount() == len(rows)


def test_panels_hidden_on_start_page_and_shown_with_an_image(make_window, image_dir):
    from PySide6.QtWidgets import QDockWidget

    win = make_window(Settings())
    assert all(not d.isVisible() for d in win.docks)
    win.open_path(image_dir)
    visible = {d.objectName() for d in win.docks if d.isVisible()}
    assert visible == {"Labels", "Shapes", "Files"}  # Flags only when there are flags
    assert all(d.features() == QDockWidget.DockWidgetFeature.DockWidgetMovable for d in win.docks)


def test_view_menu_can_toggle_panels(make_window, image_dir):
    win = make_window(Settings(flags=["blurry"]))
    win.open_path(image_dir)
    view = next(a.menu() for a in win.menuBar().actions() if a.text() == "&View")
    panels = next(a.menu() for a in view.actions() if a.text() == "Panels")
    toggles = {a.text(): a for a in panels.actions()}
    assert set(toggles) == {"Flags", "Labels", "Shapes", "Files"}
    toggles["Files"].trigger()
    assert not next(d for d in win.docks if d.objectName() == "Files").isVisible()


def test_propagate_and_track_buttons_follow_the_image_arrows(make_window):
    from PySide6.QtWidgets import QToolBar

    win = make_window(Settings())
    (toolbar,) = win.findChildren(QToolBar)
    texts = [a.text() for a in toolbar.actions() if not a.isSeparator()]
    i = texts.index("Next Image")
    assert texts[i + 1] == "Propagate"
    assert win.video.track_action.isCheckable()  # auto-propagate, in the dropdown
    assert win.propagate_button_action.menu() is win.video.menu  # dropdown with the options


def test_only_four_menus(make_window):
    win = make_window(Settings())
    assert [a.text() for a in win.menuBar().actions()] == ["&File", "&Edit", "&View", "&Help"]


def test_right_click_selects_and_offers_shape_actions(window, ui):
    from easy_mask.core.shapes import Shape, ShapeType

    window.canvas.set_shapes([Shape("cat", ShapeType.RECTANGLE, [(10, 10), (40, 40)])])
    shown = []
    window.show_menu = lambda menu, pos: shown.append(menu)
    ui.click(20, 20, Qt.MouseButton.RightButton)
    assert window.canvas.selection == [0]
    texts = [a.text() for a in shown[0].actions()]
    assert "Label" in texts and "Propagate Selected" in texts and "Mask" in texts


def test_default_show_menu_pops_the_menu_up(make_window):
    """Regression: QMenu.exec called unbound raised TypeError, so right-click did nothing."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QMenu

    win = make_window(Settings())
    menu = QMenu(win)
    shown = []
    menu.exec = lambda pos: shown.append(pos)  # the instance's own exec: no modal loop
    win.show_menu(menu, QPoint(5, 6))
    assert shown == [QPoint(5, 6)]
