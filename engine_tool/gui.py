"""PySide6 GUI entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from .exceptions import EngineToolError
from .models import EngineJob, LoopParameters, PipelineResult, StageInput
from .pipeline import run_pipeline
from .utils import ensure_path

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
RESOURCES_ROOT = WORKSPACE_ROOT / "resources"
TEMPLATE_CANDIDATE = RESOURCES_ROOT / "[engine-name]modular_engine_crankshaft.xml"
COMPILER_CANDIDATE = RESOURCES_ROOT / "component_mod_compiler.com"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "results"


class PipelineWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str, object)

    def __init__(self, job: EngineJob) -> None:
        super().__init__()
        self._job = job

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = run_pipeline(self._job, progress=self.progress.emit)
        except EngineToolError as exc:
            self.finished.emit(False, str(exc), None)
        except Exception as exc:  # pragma: no cover - best effort safeguard
            self.finished.emit(False, repr(exc), None)
        else:
            self.finished.emit(True, "Completed", result)


class StageRow(QtWidgets.QWidget):
    def __init__(self, label: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.label = label
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._on_browse)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0.0, 3600.0)
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setValue(1.0)
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0.0, 3600.0)
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setValue(2.0)
        layout.addWidget(QtWidgets.QLabel(self.label), 0, 0)
        layout.addWidget(self.path_edit, 0, 1)
        layout.addWidget(self.browse_btn, 0, 2)
        layout.addWidget(QtWidgets.QLabel("Start (s)"), 1, 0)
        layout.addWidget(self.start_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("End (s)"), 1, 2)
        layout.addWidget(self.end_spin, 1, 3)
        layout.setColumnStretch(1, 1)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Select source for {self.label}")
        if path:
            self.path_edit.setText(path)

    def to_stage_input(self) -> StageInput:
        path_text = self.path_edit.text().strip()
        if not path_text:
            raise EngineToolError(f"Stage {self.label} is missing an input file")
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end <= start:
            raise EngineToolError(f"Stage {self.label} must have end > start")
        source_path = ensure_path(path_text)
        if not source_path.exists():
            raise EngineToolError(f"Stage {self.label} file does not exist")
        return StageInput(
            label=self.label,
            source_path=source_path,
            start_sec=float(start),
            end_sec=float(end),
        )


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stormworks Engine Audio Generator")
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker_obj: Optional[PipelineWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        form = QtWidgets.QFormLayout()
        self.engine_edit = QtWidgets.QLineEdit()
        self.template_edit = QtWidgets.QLineEdit(str(TEMPLATE_CANDIDATE) if TEMPLATE_CANDIDATE.exists() else "")
        self.template_btn = QtWidgets.QPushButton("Browse…")
        self.template_btn.clicked.connect(lambda: self._pick_file(self.template_edit, "Select XML template", "XML Files (*.xml)"))
        self.compiler_edit = QtWidgets.QLineEdit(str(COMPILER_CANDIDATE) if COMPILER_CANDIDATE.exists() else "")
        self.compiler_btn = QtWidgets.QPushButton("Browse…")
        self.compiler_btn.clicked.connect(lambda: self._pick_file(self.compiler_edit, "Select component_mod_compiler", "Executables (*.com *.exe);;All Files (*.*)"))
        self.output_edit = QtWidgets.QLineEdit(str(DEFAULT_OUTPUT))
        self.output_btn = QtWidgets.QPushButton("Choose…")
        self.output_btn.clicked.connect(self._pick_directory)
        self.ffmpeg_edit = QtWidgets.QLineEdit("ffmpeg")
        self.radius_spin = QtWidgets.QSpinBox()
        self.radius_spin.setRange(10, 200000)
        self.radius_spin.setValue(4000)
        self.quality_spin = QtWidgets.QSpinBox()
        self.quality_spin.setRange(0, 10)
        self.quality_spin.setValue(3)
        form.addRow("Engine name", self.engine_edit)
        form.addRow("Template XML", self._wrap_with_button(self.template_edit, self.template_btn))
        form.addRow("component_mod_compiler", self._wrap_with_button(self.compiler_edit, self.compiler_btn))
        form.addRow("Output directory", self._wrap_with_button(self.output_edit, self.output_btn))
        form.addRow("FFmpeg executable", self.ffmpeg_edit)
        form.addRow("Zero-cross search radius (samples)", self.radius_spin)
        form.addRow("Vorbis quality (0-10)", self.quality_spin)
        layout.addLayout(form)
        self.stage_rows = [StageRow(label) for label in ("L1", "L2", "L3", "L4")]
        stages_group = QtWidgets.QGroupBox("Loop stages")
        stages_layout = QtWidgets.QVBoxLayout(stages_group)
        for row in self.stage_rows:
            stages_layout.addWidget(row)
        self.multi_select_btn = QtWidgets.QPushButton("Multi-select sources…")
        self.multi_select_btn.clicked.connect(self._on_multi_select_sources)
        stages_layout.addWidget(self.multi_select_btn)
        layout.addWidget(stages_group)
        self.run_btn = QtWidgets.QPushButton("Generate")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

    def _wrap_with_button(self, line_edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        h_layout = QtWidgets.QHBoxLayout(widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(line_edit)
        h_layout.addWidget(button)
        return widget

    def _pick_file(self, target: QtWidgets.QLineEdit, title: str, filter_mask: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, filter=filter_mask)
        if path:
            target.setText(path)

    def _pick_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_edit.setText(path)

    def _on_multi_select_sources(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select up to 4 loop sources")
        if not paths:
            return
        count = min(len(paths), len(self.stage_rows))
        for idx in range(count):
            self.stage_rows[idx].path_edit.setText(paths[idx])
        if len(paths) > len(self.stage_rows):
            QtWidgets.QMessageBox.information(self, "Extra files ignored", "Only the first four files were assigned to L1-L4.")

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_run(self) -> None:
        try:
            job = self._build_job()
        except EngineToolError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self.log_view.clear()
        self._append_log("Starting pipeline…")
        self.run_btn.setEnabled(False)
        self._worker_thread = QtCore.QThread(self)
        self._worker_obj = PipelineWorker(job)
        self._worker_obj.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker_obj.run)
        self._worker_obj.progress.connect(self._append_log)
        self._worker_obj.finished.connect(self._on_worker_finished)
        self._worker_thread.start()

    def _on_worker_finished(self, success: bool, message: str, payload: object) -> None:
        self.run_btn.setEnabled(True)
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
        if self._worker_obj:
            self._worker_obj.deleteLater()
            self._worker_obj = None
        if success and isinstance(payload, PipelineResult):
            summary = ["Pipeline completed."]
            if payload.bin_files:
                for bin_file in payload.bin_files:
                    summary.append(f"BIN: {bin_file}")
            else:
                summary.append("Compiler produced no BIN files.")
            self._append_log("\n".join(summary))
            QtWidgets.QMessageBox.information(self, "Done", "Pipeline completed successfully.")
        else:
            self._append_log(f"Error: {message}")
            QtWidgets.QMessageBox.critical(self, "Pipeline failed", message)

    def _build_job(self) -> EngineJob:
        engine_name = self.engine_edit.text().strip()
        if not engine_name:
            raise EngineToolError("Engine name is required")
        template_text = self.template_edit.text().strip()
        compiler_text = self.compiler_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not template_text:
            raise EngineToolError("Template XML path is required")
        if not compiler_text:
            raise EngineToolError("component_mod_compiler path is required")
        if not output_text:
            raise EngineToolError("Output directory is required")
        template_path = ensure_path(template_text)
        compiler_path = ensure_path(compiler_text)
        output_dir = ensure_path(output_text)
        ffmpeg_path = self.ffmpeg_edit.text().strip() or "ffmpeg"
        if not template_path.exists():
            raise EngineToolError("Template XML path does not exist")
        if not compiler_path.exists():
            raise EngineToolError("component_mod_compiler path does not exist")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        stages = [row.to_stage_input() for row in self.stage_rows]
        params = LoopParameters(search_radius=int(self.radius_spin.value()), ogg_quality=int(self.quality_spin.value()))
        return EngineJob(
            engine_name=engine_name,
            template_path=template_path,
            compiler_path=compiler_path,
            output_dir=output_dir,
            ffmpeg_path=ffmpeg_path,
            loop_params=params,
            stages=stages,
        )


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(960, 720)
    window.show()
    app.exec()


if __name__ == "__main__":  # pragma: no cover
    main()
