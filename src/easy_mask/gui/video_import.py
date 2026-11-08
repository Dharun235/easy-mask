"""Import a video as images: pick format, folder and frame rate, then extract in the background.

Everything in easy-mask works on folders of images, so a video is first turned into one.
The dialog shows what the import will produce before it starts ("625 PNG images"), and
the worker reports progress so long videos never look frozen.
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from easy_mask.io.video import (
    IMAGE_FORMATS,
    JPEG_QUALITY,
    VIDEO_EXTENSIONS,
    VideoInfo,
    extract_frames,
    frames_dir_for,
    planned_count,
)


@dataclass
class ImportOptions:
    """Everything a video import needs."""

    video: Path
    out_dir: Path
    fmt: str = "jpg"
    quality: int = JPEG_QUALITY
    step: int = 1
    max_frames: int | None = None
    open_after: bool = True


class VideoImportDialog(QDialog):
    """Choose output folder, image format / quality, every n-th frame and a frame limit."""

    def __init__(self, defaults: ImportOptions, info: VideoInfo, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import Video Frames")
        self.info = info
        self.video = defaults.video

        self.out_edit = QLineEdit(str(defaults.out_dir))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(browse)

        self.format_box = QComboBox()
        for key, name in IMAGE_FORMATS.items():
            self.format_box.addItem(name, key)
        self.format_box.setCurrentIndex(list(IMAGE_FORMATS).index(defaults.fmt))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(defaults.quality)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(defaults.step)
        self.step_spin.setSuffix(" (1 = every frame)")
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 1_000_000)
        self.max_spin.setValue(defaults.max_frames or 0)
        self.max_spin.setSpecialValueText("all")
        self.open_box = QCheckBox("Open the images when done")
        self.open_box.setChecked(defaults.open_after)
        self.summary = QLabel()

        form = QFormLayout()
        form.addRow("Video", QLabel(f"{self.video.name}  ({_describe(info)})"))
        form.addRow("Save images to", out_row)
        form.addRow("Format", self.format_box)
        form.addRow("Quality (JPEG / WebP)", self.quality_spin)
        form.addRow("Keep every n-th frame", self.step_spin)
        form.addRow("At most", self.max_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.summary)
        layout.addWidget(self.open_box)
        layout.addWidget(buttons)

        for signal in (
            self.format_box.currentIndexChanged,
            self.step_spin.valueChanged,
            self.max_spin.valueChanged,
        ):
            signal.connect(self._refresh)
        self._refresh()

    def options(self) -> ImportOptions:
        """The choices currently in the dialog."""
        return ImportOptions(
            video=self.video,
            out_dir=Path(self.out_edit.text()).expanduser(),
            fmt=self.format_box.currentData(),
            quality=self.quality_spin.value(),
            step=self.step_spin.value(),
            max_frames=self.max_spin.value() or None,
            open_after=self.open_box.isChecked(),
        )

    def _refresh(self) -> None:
        opts = self.options()
        self.quality_spin.setEnabled(opts.fmt != "png")
        count = planned_count(self.info.frames, opts.step, opts.max_frames)
        what = f"{count} {opts.fmt.upper()} images" if count else f"{opts.fmt.upper()} images"
        self.summary.setText(f"Will write {what}.")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Save images to", self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    @staticmethod
    def ask(
        defaults: ImportOptions, info: VideoInfo, parent: QWidget | None = None
    ) -> ImportOptions | None:
        """Show the dialog modally; the chosen options, or None if cancelled."""
        dialog = VideoImportDialog(defaults, info, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.options()
        return None


def _describe(info: VideoInfo) -> str:
    seconds = f", {info.seconds:.0f} s" if info.seconds else ""
    fps = f", {info.fps:.0f} fps" if info.fps else ""
    return f"{info.frames} frames{fps}{seconds}, {info.width}x{info.height}"


def default_options(video: Path, step: int = 1, max_frames: int | None = None) -> ImportOptions:
    """Sensible defaults: JPEG next to the video in ``<name>_frames``."""
    return ImportOptions(video, frames_dir_for(video), step=step, max_frames=max_frames)


VIDEO_FILTER = "Videos (" + " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS)) + ")"


class VideoImportWorker(QThread):
    """Extracts frames on a background thread.

    Signals:
        progress(int, int): Images written, images planned (0 = unknown).
        failed(str): Extraction failed; the message is meant for the user.
        done(object): Finished; the output folder (a Path).
    """

    progress = Signal(int, int)
    failed = Signal(str)
    done = Signal(object)

    def __init__(self, options: ImportOptions):
        super().__init__()
        self.options = options
        self._cancelled = False

    def cancel(self) -> None:
        """Stop after the current frame; images written so far are kept."""
        self._cancelled = True

    def run(self) -> None:
        """Runs on the worker thread."""
        o = self.options
        try:
            folder = extract_frames(
                o.video,
                o.out_dir,
                step=o.step,
                max_frames=o.max_frames,
                fmt=o.fmt,
                quality=o.quality,
                progress=self.progress.emit,
                should_stop=lambda: self._cancelled,
            )
        except (ValueError, OSError) as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(folder)
