"""The real label dialog, driven with keys and clicks (no stubs)."""

from PySide6.QtCore import Qt

from easy_mask.gui.label_dialog import LabelDialog, LabelInfo


def make(qtbot, initial=None):
    dialog = LabelDialog(["cat", "dog", "bird"], initial or LabelInfo(""))
    qtbot.addWidget(dialog)
    return dialog


def test_typed_values_are_returned(qtbot):
    dialog = make(qtbot)
    qtbot.keyClicks(dialog.label_edit, "  zebra ")
    qtbot.keyClicks(dialog.group_edit, "7")
    qtbot.keyClicks(dialog.description_edit, "striped")
    assert dialog.result_info() == LabelInfo("zebra", 7, "striped")


def test_group_id_rejects_letters(qtbot):
    dialog = make(qtbot)
    qtbot.keyClicks(dialog.group_edit, "ab12")
    assert dialog.result_info().group_id == 12


def test_empty_label_cannot_be_accepted(qtbot):
    dialog = make(qtbot)
    dialog._accept_if_valid()
    assert dialog.result() != dialog.DialogCode.Accepted


def test_click_fills_and_double_click_accepts(qtbot):
    dialog = make(qtbot)
    item = dialog.label_list.item(1)
    dialog.label_list.itemClicked.emit(item)
    assert dialog.label_edit.text() == "dog"
    with qtbot.waitSignal(dialog.accepted):
        dialog.label_list.itemDoubleClicked.emit(dialog.label_list.item(2))
    assert dialog.result_info().label == "bird"


def test_initial_values_prefilled(qtbot):
    dialog = make(qtbot, LabelInfo("cat", 3, "note"))
    assert (dialog.label_edit.text(), dialog.group_edit.text()) == ("cat", "3")
    assert dialog.description_edit.text() == "note"


def test_completer_matches_substring_case_insensitive(qtbot):
    dialog = make(qtbot)
    completer = dialog.label_edit.completer()
    completer.setCompletionPrefix("IR")
    assert completer.currentCompletion() == "bird"
    assert completer.caseSensitivity() == Qt.CaseSensitivity.CaseInsensitive


def test_strict_dialog_offers_and_accepts_only_allowed_labels(qtbot):
    dialog = LabelDialog(["cat", "zebra"], LabelInfo(""), allowed=["cat", "dog"])
    qtbot.addWidget(dialog)
    assert [dialog.label_list.item(i).text() for i in range(dialog.label_list.count())] == [
        "cat",
        "dog",
    ]
    qtbot.keyClicks(dialog.label_edit, "zebra")
    dialog._accept_if_valid()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert "not in the label list" in dialog.error_label.text()
    dialog.label_edit.setText("dog")
    with qtbot.waitSignal(dialog.accepted):
        dialog._accept_if_valid()


def test_ask_returns_none_when_cancelled(qtbot, monkeypatch):
    monkeypatch.setattr(LabelDialog, "exec", lambda self: LabelDialog.DialogCode.Rejected)
    assert LabelDialog.ask(["cat"], LabelInfo("cat")) is None
    monkeypatch.setattr(LabelDialog, "exec", lambda self: LabelDialog.DialogCode.Accepted)
    assert LabelDialog.ask(["cat"], LabelInfo("cat")) == LabelInfo("cat")
