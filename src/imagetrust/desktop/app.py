"""
ImageTrust Desktop Application (PySide6/Qt).

Modern, cross-platform desktop GUI for AI image forensics.
Features:
- Drag-and-drop image loading
- Real-time analysis with progress feedback
- Detailed forensic reports
- Configurable analysis settings
- Export results to JSON/PDF
- Keyboard shortcuts

Designed for Windows .exe packaging with PyInstaller.
"""

import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import Qt, QThread, Signal, QSize, QMimeData
    from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QAction, QKeySequence, QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar,
        QCheckBox, QGroupBox, QTextEdit, QSplitter, QFrame, QScrollArea,
        QGridLayout, QStatusBar, QMenuBar, QMenu, QSizePolicy, QSpacerItem,
    )
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

from PIL import Image


# Style constants
DARK_BG = "#0f1117"
DARK_SURFACE = "#1a1d24"
DARK_BORDER = "#2d3139"
TEXT_PRIMARY = "#e5e7eb"
TEXT_SECONDARY = "#9ca3af"
ACCENT_COLOR = "#6366f1"
ACCENT_HOVER = "#4f46e5"
SUCCESS_COLOR = "#22c55e"
ERROR_COLOR = "#ef4444"
WARNING_COLOR = "#f59e0b"


STYLESHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}

QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}}

QLabel {{
    color: {TEXT_PRIMARY};
}}

QLabel#title {{
    font-size: 24px;
    font-weight: bold;
    color: white;
}}

QLabel#subtitle {{
    font-size: 13px;
    color: {TEXT_SECONDARY};
}}

QLabel#section {{
    font-size: 15px;
    font-weight: bold;
    color: {TEXT_PRIMARY};
    padding: 8px 0;
}}

QLabel#verdict {{
    font-size: 18px;
    font-weight: bold;
}}

QPushButton {{
    background-color: {DARK_SURFACE};
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    padding: 10px 20px;
    color: {TEXT_PRIMARY};
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {DARK_BORDER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_COLOR};
}}

QPushButton#accent {{
    background-color: {ACCENT_COLOR};
    border: none;
    color: white;
}}

QPushButton#accent:hover {{
    background-color: {ACCENT_HOVER};
}}

QGroupBox {{
    background-color: {DARK_SURFACE};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: {TEXT_PRIMARY};
}}

QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {DARK_BORDER};
    background-color: {DARK_BG};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT_COLOR};
    border-color: {ACCENT_COLOR};
}}

QTextEdit {{
    background-color: {DARK_BG};
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    padding: 12px;
    color: {TEXT_PRIMARY};
    font-family: "Consolas", "SF Mono", monospace;
    font-size: 12px;
}}

QProgressBar {{
    background-color: {DARK_BORDER};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_COLOR};
    border-radius: 4px;
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QStatusBar {{
    background-color: {DARK_SURFACE};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {DARK_BORDER};
}}

QMenuBar {{
    background-color: {DARK_SURFACE};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {DARK_BORDER};
}}

QMenuBar::item:selected {{
    background-color: {DARK_BORDER};
}}

QMenu {{
    background-color: {DARK_SURFACE};
    border: 1px solid {DARK_BORDER};
}}

QMenu::item:selected {{
    background-color: {ACCENT_COLOR};
}}

QSplitter::handle {{
    background-color: {DARK_BORDER};
}}
"""


class AnalysisWorker(QThread):
    """Background worker for image analysis."""

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, image: Image.Image, image_bytes: bytes, image_path: str, settings: dict):
        super().__init__()
        self.image = image
        self.image_bytes = image_bytes
        self.image_path = image_path
        self.settings = settings

    def run(self):
        try:
            from types import SimpleNamespace
            from imagetrust.detection.multi_detector import ComprehensiveDetector
            from imagetrust.utils.scoring import analyze_image_source, compute_combined_score

            self.progress.emit(10, "Loading detector...")

            detector = ComprehensiveDetector()

            self.progress.emit(30, "Running AI detection models...")

            result = detector.analyze(self.image)

            self.progress.emit(60, "Computing scores...")

            # Extract metadata
            metadata = {"has_exif": False}
            try:
                import exifread
                from io import BytesIO
                tags = exifread.process_file(BytesIO(self.image_bytes), details=False)
                metadata["has_exif"] = bool(tags)
            except Exception:
                pass

            uploaded_file = SimpleNamespace(name=self.image_path or "image.jpg")
            source_info = analyze_image_source(self.image, self.image_bytes, uploaded_file, metadata)

            self.progress.emit(80, "Finalizing analysis...")

            score_info = compute_combined_score(
                result,
                uploaded_file,
                source_info=source_info,
                settings=self.settings,
            )

            self.progress.emit(100, "Complete!")

            # Compute hash
            hash_hex = hashlib.sha256(self.image_bytes).hexdigest()

            self.finished.emit({
                "score_info": score_info,
                "source_info": source_info,
                "metadata": metadata,
                "hash": hash_hex,
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            self.error.emit(str(e))


class DropZone(QLabel):
    """Image drop zone with drag-and-drop support."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DARK_SURFACE};
                border: 2px dashed {DARK_BORDER};
                border-radius: 12px;
                color: {TEXT_SECONDARY};
            }}
        """)
        self._set_placeholder()

    def _set_placeholder(self):
        self.setText("Drop image here\nor click 'Open Image'")

    def set_image(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QLabel {{
                    background-color: {DARK_SURFACE};
                    border: 2px dashed {ACCENT_COLOR};
                    border-radius: 12px;
                }}
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DARK_SURFACE};
                border: 2px dashed {DARK_BORDER};
                border-radius: 12px;
                color: {TEXT_SECONDARY};
            }}
        """)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                self.file_dropped.emit(path)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DARK_SURFACE};
                border: 2px dashed {DARK_BORDER};
                border-radius: 12px;
            }}
        """)


class ImageTrustApp(QMainWindow):
    """Main ImageTrust Desktop Application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageTrust Desktop")
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)

        # State
        self.image: Optional[Image.Image] = None
        self.image_bytes: Optional[bytes] = None
        self.image_path: Optional[str] = None
        self.worker: Optional[AnalysisWorker] = None
        self.last_result: Optional[dict] = None

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()

    def _setup_ui(self):
        """Build the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(f"background-color: {DARK_SURFACE}; border-bottom: 1px solid {DARK_BORDER};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        title = QLabel("ImageTrust Desktop")
        title.setObjectName("title")
        title_layout.addWidget(title)

        subtitle = QLabel("AI Image Forensics - Offline Analysis")
        subtitle.setObjectName("subtitle")
        title_layout.addWidget(subtitle)

        header_layout.addWidget(title_container)
        header_layout.addStretch()

        # Version label
        version_label = QLabel("v1.0.0")
        version_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        header_layout.addWidget(version_label)

        main_layout.addWidget(header)

        # Content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setContentsMargins(16, 16, 16, 16)

        # Left panel - Image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 8, 16)

        section_label = QLabel("Image")
        section_label.setObjectName("section")
        left_layout.addWidget(section_label)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self._load_image_path)
        left_layout.addWidget(self.drop_zone, 1)

        btn_container = QHBoxLayout()
        self.open_btn = QPushButton("Open Image")
        self.open_btn.setObjectName("accent")
        self.open_btn.clicked.connect(self._open_file_dialog)
        btn_container.addWidget(self.open_btn)
        btn_container.addStretch()
        left_layout.addLayout(btn_container)

        self.file_label = QLabel("No image selected")
        self.file_label.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 8px 0;")
        left_layout.addWidget(self.file_label)

        splitter.addWidget(left_panel)

        # Right panel - Analysis
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 16, 16, 16)

        section_label2 = QLabel("Analysis")
        section_label2.setObjectName("section")
        right_layout.addWidget(section_label2)

        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout(settings_group)

        self.use_ml = QCheckBox("AI Detection Models")
        self.use_ml.setChecked(True)
        self.use_frequency = QCheckBox("Frequency Analysis")
        self.use_noise = QCheckBox("Noise Pattern Analysis")
        self.auto_calibration = QCheckBox("Auto Calibration")
        self.auto_calibration.setChecked(True)
        self.high_confidence = QCheckBox("High Confidence Mode (>=75%)")
        self.high_confidence.setChecked(True)

        settings_layout.addWidget(self.use_ml, 0, 0)
        settings_layout.addWidget(self.use_frequency, 0, 1)
        settings_layout.addWidget(self.use_noise, 1, 0)
        settings_layout.addWidget(self.auto_calibration, 1, 1)
        settings_layout.addWidget(self.high_confidence, 2, 0)

        right_layout.addWidget(settings_group)

        # Analyze button
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.setObjectName("accent")
        self.analyze_btn.setFixedHeight(44)
        self.analyze_btn.clicked.connect(self._run_analysis)
        right_layout.addWidget(self.analyze_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self.progress_label.setVisible(False)
        right_layout.addWidget(self.progress_label)

        # Results group
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.verdict_label = QLabel("Verdict: -")
        self.verdict_label.setObjectName("verdict")
        results_layout.addWidget(self.verdict_label)

        self.prob_label = QLabel("AI Probability: - | Real Probability: -")
        results_layout.addWidget(self.prob_label)

        self.confidence_label = QLabel("Confidence: -")
        results_layout.addWidget(self.confidence_label)

        self.ai_progress = QProgressBar()
        self.ai_progress.setMaximum(100)
        self.ai_progress.setValue(0)
        results_layout.addWidget(self.ai_progress)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-style: italic;")
        results_layout.addWidget(self.note_label)

        right_layout.addWidget(results_group)

        # Details group
        details_group = QGroupBox("Detailed Report")
        details_layout = QVBoxLayout(details_group)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Analysis results will appear here...")
        details_layout.addWidget(self.details_text)

        right_layout.addWidget(details_group, 1)

        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        right_layout.addLayout(export_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([500, 700])

        main_layout.addWidget(splitter, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        export_action = QAction("Export Report...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_report)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        run_action = QAction("Run Analysis", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self._run_analysis)
        analysis_menu.addAction(run_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About ImageTrust", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        pass  # Already handled in menu actions

    def _open_file_dialog(self):
        """Open file dialog to select image."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp);;All Files (*)"
        )
        if path:
            self._load_image_path(path)

    def _load_image_path(self, path: str):
        """Load image from path."""
        try:
            with open(path, "rb") as f:
                image_bytes = f.read()

            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            self.image = image
            self.image_bytes = image_bytes
            self.image_path = path

            # Display preview
            qimage = self._pil_to_qimage(image)
            pixmap = QPixmap.fromImage(qimage)
            self.drop_zone.set_image(pixmap)

            # Update file label
            file_size = len(image_bytes) / 1024
            self.file_label.setText(
                f"{os.path.basename(path)} | {image.width}x{image.height} | {file_size:.1f} KB"
            )

            self.status_bar.showMessage(f"Loaded: {path}")

            # Reset results
            self._reset_results()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def _pil_to_qimage(self, pil_image: Image.Image) -> QImage:
        """Convert PIL Image to QImage."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        return QImage(
            data,
            pil_image.width,
            pil_image.height,
            pil_image.width * 3,
            QImage.Format_RGB888
        )

    def _reset_results(self):
        """Reset analysis results."""
        self.verdict_label.setText("Verdict: -")
        self.verdict_label.setStyleSheet("")
        self.prob_label.setText("AI Probability: - | Real Probability: -")
        self.confidence_label.setText("Confidence: -")
        self.ai_progress.setValue(0)
        self.note_label.setText("")
        self.details_text.clear()
        self.export_btn.setEnabled(False)
        self.last_result = None

    def _run_analysis(self):
        """Start analysis in background thread."""
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        if self.worker is not None and self.worker.isRunning():
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.analyze_btn.setEnabled(False)

        settings = {
            "use_ml": self.use_ml.isChecked(),
            "use_frequency": self.use_frequency.isChecked(),
            "use_noise": self.use_noise.isChecked(),
            "auto_calibration": self.auto_calibration.isChecked(),
        }

        self.worker = AnalysisWorker(
            self.image, self.image_bytes, self.image_path, settings
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_analysis_complete)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.start()

    def _on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_analysis_complete(self, result: dict):
        """Handle analysis completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.last_result = result

        score_info = result["score_info"]
        combined = score_info["combined"]
        ai_prob = score_info["ai_prob"]
        real_prob = score_info["real_prob"]

        # Determine verdict
        if self.high_confidence.isChecked():
            real_threshold = 0.25
            ai_threshold = 0.75
        else:
            real_threshold = 0.35
            ai_threshold = 0.60

        if combined >= ai_threshold:
            verdict = "AI Generated"
            color = ERROR_COLOR
        elif combined <= real_threshold:
            verdict = "Real Photograph"
            color = SUCCESS_COLOR
        else:
            verdict = "Inconclusive"
            color = WARNING_COLOR

        self.verdict_label.setText(f"Verdict: {verdict}")
        self.verdict_label.setStyleSheet(f"color: {color};")

        self.prob_label.setText(f"AI Probability: {ai_prob:.1%} | Real Probability: {real_prob:.1%}")
        self.confidence_label.setText(f"Confidence: {max(ai_prob, real_prob):.1%}")
        self.ai_progress.setValue(int(ai_prob * 100))

        # Notes
        notes = []
        if score_info.get("calibration_applied"):
            notes.append(score_info.get("calibration_note", "Auto calibration applied."))
        if score_info.get("signal_suppressed"):
            notes.append("Signal analysis suppressed due to compression.")
        self.note_label.setText(" ".join(notes))

        # Detailed report
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("IMAGETRUST FORENSIC REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"\nVerdict: {verdict}")
        report_lines.append(f"AI Probability: {ai_prob:.1%}")
        report_lines.append(f"Real Probability: {real_prob:.1%}")
        report_lines.append(f"Combined Score: {combined:.3f}")
        report_lines.append(f"\nTimestamp: {result['timestamp']}")
        report_lines.append(f"SHA256: {result['hash']}")

        report_lines.append("\n" + "-" * 50)
        report_lines.append("Active Methods:")
        for r in score_info.get("active_results", []):
            report_lines.append(f"  - {r['method']}: {r['ai_probability']:.1%} AI")

        report_lines.append("\n" + "-" * 50)
        report_lines.append("Weights:")
        weights = score_info.get("weights", {})
        report_lines.append(f"  ML: {weights.get('ml', 0)}")
        report_lines.append(f"  Signal: {weights.get('signal', 0)}")
        report_lines.append(f"  Vote: {weights.get('vote', 0)}")

        source_info = result.get("source_info", {})
        report_lines.append("\n" + "-" * 50)
        report_lines.append("Image Source Analysis:")
        report_lines.append(f"  Platform: {source_info.get('platform', 'Unknown')}")
        report_lines.append(f"  Compression: {source_info.get('compression_level', 'Unknown')}")
        report_lines.append(f"  BPP: {source_info.get('bpp', 0):.2f}")

        self.details_text.setText("\n".join(report_lines))
        self.export_btn.setEnabled(True)

        self.status_bar.showMessage(f"Analysis complete: {verdict}")

    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.analyze_btn.setEnabled(True)

        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error}")
        self.status_bar.showMessage("Analysis failed")

    def _export_report(self):
        """Export analysis report to JSON."""
        if self.last_result is None:
            QMessageBox.warning(self, "No Results", "Run analysis first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            f"imagetrust_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )

        if path:
            try:
                export_data = {
                    "version": "1.0.0",
                    "timestamp": self.last_result["timestamp"],
                    "image": {
                        "path": self.image_path,
                        "width": self.image.width,
                        "height": self.image.height,
                        "sha256": self.last_result["hash"],
                    },
                    "analysis": self.last_result["score_info"],
                    "source_info": self.last_result["source_info"],
                }

                with open(path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

                self.status_bar.showMessage(f"Report exported: {path}")
                QMessageBox.information(self, "Export Complete", f"Report saved to:\n{path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ImageTrust",
            "ImageTrust Desktop v1.0.0\n\n"
            "AI Image Forensics Tool\n\n"
            "Detect AI-generated images using ensemble\n"
            "of machine learning models and signal analysis.\n\n"
            "Built for Master's Thesis\n"
            "2024-2025"
        )


def main():
    """Entry point for desktop application."""
    if not PYSIDE6_AVAILABLE:
        print("Error: PySide6 is not installed.")
        print("Install with: pip install 'imagetrust[desktop]'")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = ImageTrustApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
