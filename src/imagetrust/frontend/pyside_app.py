"""
ImageTrust Desktop Application (PySide6 / Qt6)

Professional offline AI image forensics desktop application.
Integrates ComprehensiveDetector with empirically calibrated thresholds.

Run directly:  python -m imagetrust.frontend.pyside_app
Or:             python src/imagetrust/frontend/pyside_app.py
"""

# ---------------------------------------------------------------------------
# Import order is critical to avoid shiboken6 / six conflict.
#
# shiboken6 (shipped with PySide6) patches builtins.__import__ with a
# "feature_import" wrapper.  When the `six` package later installs its
# _SixMetaPathImporter on sys.meta_path, shiboken's wrapper tries to
# access importer._path which does not exist -> AttributeError.
#
# Fix: pre-import `six` and its lazy-loaded `moves` sub-modules BEFORE
# PySide6 so they are already cached in sys.modules and the meta-path
# importer is never invoked under shiboken's patched __import__.
# ---------------------------------------------------------------------------
import sys as _sys  # noqa: E402

try:
    import six  # noqa: F401, E402
    import six.moves  # noqa: F401, E402
    import six.moves.urllib  # noqa: F401, E402
    import six.moves.urllib.parse  # noqa: F401, E402
    import six.moves.urllib.request  # noqa: F401, E402
except ImportError:
    pass

# Also pre-import dateutil which is the main trigger of the six conflict
try:
    import dateutil  # noqa: F401, E402
    import dateutil.parser  # noqa: F401, E402
    import dateutil.tz  # noqa: F401, E402
except ImportError:
    pass

from PySide6.QtCore import (  # noqa: E402
    Qt,
    QSize,
    QThread,
    Signal,
    QMimeData,
    QTimer,
)
from PySide6.QtGui import (  # noqa: E402
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import io
import json
import os
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
_COLORS = {
    "bg_primary": "#0F1117",
    "bg_secondary": "#161B22",
    "bg_card": "#1C2128",
    "bg_input": "#21262D",
    "border": "#30363D",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted": "#6E7681",
    "accent": "#4F46E5",
    "accent_hover": "#4338CA",
    "success": "#22C55E",
    "danger": "#EF4444",
    "warning": "#F59E0B",
    "info": "#3B82F6",
}

_STYLESHEET = f"""
QMainWindow {{
    background-color: {_COLORS['bg_primary']};
}}
QWidget {{
    background-color: {_COLORS['bg_primary']};
    color: {_COLORS['text_primary']};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {_COLORS['bg_card']};
    border: 1px solid {_COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {_COLORS['text_primary']};
}}
QLabel {{
    background: transparent;
    border: none;
}}
QPushButton {{
    background-color: {_COLORS['bg_input']};
    color: {_COLORS['text_primary']};
    border: 1px solid {_COLORS['border']};
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {_COLORS['border']};
}}
QPushButton:pressed {{
    background-color: {_COLORS['accent']};
}}
QPushButton#accentBtn {{
    background-color: {_COLORS['accent']};
    color: #FFFFFF;
    border: none;
    font-size: 14px;
    padding: 10px 28px;
}}
QPushButton#accentBtn:hover {{
    background-color: {_COLORS['accent_hover']};
}}
QPushButton#accentBtn:disabled {{
    background-color: {_COLORS['bg_input']};
    color: {_COLORS['text_muted']};
}}
QCheckBox {{
    spacing: 8px;
    background: transparent;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid {_COLORS['border']};
    background-color: {_COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {_COLORS['accent']};
    border-color: {_COLORS['accent']};
}}
QProgressBar {{
    background-color: {_COLORS['bg_input']};
    border: 1px solid {_COLORS['border']};
    border-radius: 4px;
    height: 18px;
    text-align: center;
    color: {_COLORS['text_primary']};
    font-size: 11px;
}}
QProgressBar::chunk {{
    border-radius: 3px;
}}
QTableWidget {{
    background-color: {_COLORS['bg_card']};
    border: 1px solid {_COLORS['border']};
    border-radius: 6px;
    gridline-color: {_COLORS['border']};
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QHeaderView::section {{
    background-color: {_COLORS['bg_secondary']};
    color: {_COLORS['text_secondary']};
    border: 1px solid {_COLORS['border']};
    padding: 6px 8px;
    font-weight: bold;
    font-size: 11px;
}}
QTextEdit {{
    background-color: {_COLORS['bg_card']};
    color: {_COLORS['text_primary']};
    border: 1px solid {_COLORS['border']};
    border-radius: 6px;
    padding: 8px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {_COLORS['bg_secondary']};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {_COLORS['border']};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QStatusBar {{
    background-color: {_COLORS['bg_secondary']};
    color: {_COLORS['text_secondary']};
    border-top: 1px solid {_COLORS['border']};
    font-size: 12px;
}}
QSplitter::handle {{
    background-color: {_COLORS['border']};
    width: 2px;
}}
"""


# ---------------------------------------------------------------------------
# Worker thread for non-blocking analysis
# ---------------------------------------------------------------------------
class AnalysisWorker(QThread):
    """Runs detector init + analysis in a background thread (non-blocking)."""

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    detector_ready = Signal(object)  # passes detector back to main thread

    def __init__(self, detector, image: Image.Image, image_bytes: bytes,
                 image_path: str, settings: dict, kaggle_path: str = None):
        super().__init__()
        self.detector = detector
        self.image = image
        self.image_bytes = image_bytes
        self.image_path = image_path
        self.settings = settings
        self.kaggle_path = kaggle_path

    def run(self):
        try:
            # --- Lazy detector init (runs here, NOT on UI thread) ---
            if self.detector is None:
                self.progress.emit("Loading AI detection models (first run)...")

                # Patch _SixMetaPathImporter if needed (shiboken6 conflict)
                for importer in sys.meta_path:
                    if (type(importer).__name__ == "_SixMetaPathImporter"
                            and not hasattr(importer, "_path")):
                        importer._path = None

                from imagetrust.detection.multi_detector import (
                    ComprehensiveDetector,
                )

                self.detector = ComprehensiveDetector(
                    kaggle_model_path=self.kaggle_path
                )
                self.detector_ready.emit(self.detector)

            # --- Run analysis ---
            self.progress.emit("Running AI detection models...")
            t0 = time.perf_counter()

            result = self.detector.analyze(
                self.image,
                return_uncertainty=True,
                profile=True,
            )

            elapsed = time.perf_counter() - t0

            # Compute combined score
            self.progress.emit("Computing combined score...")
            from imagetrust.utils.scoring import (
                analyze_image_source,
                compute_combined_score,
            )

            metadata = _extract_metadata(self.image_bytes)
            uploaded = SimpleNamespace(name=self.image_path)
            source_info = analyze_image_source(
                self.image, self.image_bytes, uploaded, metadata
            )
            score_info = compute_combined_score(
                result, uploaded,
                source_info=source_info,
                settings=self.settings,
            )

            # Merge everything
            result["score_info"] = score_info
            result["source_info"] = source_info
            result["metadata"] = metadata
            result["elapsed_s"] = elapsed
            result["sha256"] = hashlib.sha256(self.image_bytes).hexdigest()
            result["report_time"] = datetime.now().isoformat()

            # --- Forensics analysis (screenshot, platform, compression) ---
            self.progress.emit("Running forensic analysis...")
            try:
                from imagetrust.forensics.engine import ForensicsEngine

                forensics = ForensicsEngine()
                forensics_report = forensics.analyze(self.image)
                result["forensics"] = forensics_report.verdict.to_dict()
                result["forensics_results"] = [
                    {
                        "plugin": r.plugin_name,
                        "category": r.category.value if hasattr(r.category, "value") else str(r.category),
                        "score": r.score,
                        "detected": r.detected,
                        "explanation": r.explanation,
                    }
                    for r in forensics_report.results
                ]
            except Exception as e:
                result["forensics"] = None
                result["forensics_results"] = []
                import traceback
                logger_msg = f"Forensics analysis failed: {e}\n{traceback.format_exc()}"
                # Don't fail the whole analysis for forensics errors
                result["forensics_error"] = str(e)

            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


def _extract_metadata(image_bytes: bytes) -> dict:
    """Minimal EXIF extraction."""
    meta = {"has_exif": False, "tags": {}}
    try:
        import exifread
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        meta["has_exif"] = bool(tags)
        meta["tags"] = {k: str(v) for k, v in tags.items()}
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------
class DropZone(QLabel):
    """Image drop zone with drag-and-drop support."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._set_placeholder()

    def _set_placeholder(self):
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {_COLORS['bg_card']};
                border: 2px dashed {_COLORS['border']};
                border-radius: 12px;
                color: {_COLORS['text_secondary']};
                font-size: 14px;
            }}
        """)
        self.setText("Drag & Drop Image Here\nor click 'Open Image' below")

    def set_image(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {_COLORS['bg_card']};
                border: 1px solid {_COLORS['border']};
                border-radius: 12px;
                padding: 8px;
            }}
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                QLabel {{
                    background-color: {_COLORS['bg_card']};
                    border: 2px solid {_COLORS['accent']};
                    border-radius: 12px;
                    color: {_COLORS['accent']};
                    font-size: 14px;
                }}
            """)

    def dragLeaveEvent(self, event):
        if not self.pixmap():
            self._set_placeholder()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
                self.file_dropped.emit(path)
            else:
                self._set_placeholder()
                self.setText("Unsupported file format.\nUse JPG, PNG, or WebP.")


class VerdictBanner(QFrame):
    """Large colour-coded verdict display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_COLORS['bg_card']};
                border: 1px solid {_COLORS['border']};
                border-radius: 10px;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        self._icon = QLabel()
        self._icon.setFixedSize(36, 36)
        self._icon.setAlignment(Qt.AlignCenter)
        self._icon.setStyleSheet("font-size: 28px;")
        layout.addWidget(self._icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        self._verdict = QLabel("Awaiting Analysis")
        self._verdict.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {_COLORS['text_secondary']};"
        )
        self._subtitle = QLabel("Load an image and click Analyze")
        self._subtitle.setStyleSheet(
            f"font-size: 12px; color: {_COLORS['text_muted']};"
        )
        text_col.addWidget(self._verdict)
        text_col.addWidget(self._subtitle)
        layout.addLayout(text_col, 1)

        self._prob_label = QLabel("")
        self._prob_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._prob_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self._prob_label)

    def set_verdict(self, verdict: str, ai_prob: float, subtitle: str = ""):
        if verdict == "ai_generated":
            color = _COLORS["danger"]
            icon_text = "!"
            label = "AI-Generated"
        elif verdict == "real":
            color = _COLORS["success"]
            icon_text = "OK"
            label = "Real Photograph"
        else:
            color = _COLORS["warning"]
            icon_text = "?"
            label = "Uncertain"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_COLORS['bg_card']};
                border: 2px solid {color};
                border-radius: 10px;
            }}
        """)
        self._icon.setText(icon_text)
        self._icon.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {color}; "
            f"background: transparent; border: 2px solid {color}; "
            f"border-radius: 18px;"
        )
        self._verdict.setText(label)
        self._verdict.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )
        self._subtitle.setText(subtitle)
        self._prob_label.setText(f"{ai_prob:.1%}")
        self._prob_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {color};"
        )

    def reset(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_COLORS['bg_card']};
                border: 1px solid {_COLORS['border']};
                border-radius: 10px;
            }}
        """)
        self._icon.setText("")
        self._icon.setStyleSheet("font-size: 28px;")
        self._verdict.setText("Awaiting Analysis")
        self._verdict.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {_COLORS['text_secondary']};"
        )
        self._subtitle.setText("Load an image and click Analyze")
        self._prob_label.setText("")


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class ImageTrustWindow(QMainWindow):
    """Professional forensic analysis desktop application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageTrust - AI Image Forensics")
        self.setMinimumSize(1100, 740)
        self.resize(1360, 860)

        self._image: Optional[Image.Image] = None
        self._image_bytes: Optional[bytes] = None
        self._image_path: Optional[str] = None
        self._result: Optional[Dict[str, Any]] = None
        self._worker: Optional[AnalysisWorker] = None
        self._detector = None  # lazy-loaded

        self._build_ui()
        self._setup_statusbar()
        self.statusBar().showMessage("Ready  |  Load an image to begin")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        header = self._build_header()
        root_layout.addWidget(header)

        # Main content (splitter)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = self._build_left_panel()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_widget = self._build_right_panel()
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        splitter.setSizes([440, 900])
        root_layout.addWidget(splitter, 1)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {_COLORS['bg_secondary']};
                border-bottom: 1px solid {_COLORS['border']};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("ImageTrust")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {_COLORS['text_primary']}; "
            "background: transparent;"
        )
        h_layout.addWidget(title)

        subtitle = QLabel("AI Image Forensics Desktop")
        subtitle.setStyleSheet(
            f"font-size: 13px; color: {_COLORS['text_muted']}; background: transparent;"
        )
        h_layout.addWidget(subtitle)
        h_layout.addStretch()

        self._device_label = QLabel("")
        self._device_label.setStyleSheet(
            f"font-size: 11px; color: {_COLORS['text_muted']}; background: transparent;"
        )
        h_layout.addWidget(self._device_label)
        return header

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 8, 16)
        layout.setSpacing(12)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._load_image)
        layout.addWidget(self._drop_zone, 1)

        # Open button
        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open Image")
        open_btn.setObjectName("accentBtn")
        open_btn.clicked.connect(self._open_file_dialog)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Image details group
        details_group = QGroupBox("Image Details")
        details_layout = QGridLayout(details_group)
        details_layout.setSpacing(6)

        self._detail_labels = {}
        fields = [
            ("File", "file"), ("Format", "format"), ("Dimensions", "dims"),
            ("File Size", "size"), ("EXIF Data", "exif"),
            ("Compression", "compression"), ("Source Hint", "source"),
            ("BPP", "bpp"), ("SHA-256", "sha256"),
        ]
        for row, (label_text, key) in enumerate(fields):
            lbl = QLabel(f"{label_text}:")
            lbl.setStyleSheet(
                f"color: {_COLORS['text_secondary']}; font-size: 12px;"
            )
            val = QLabel("-")
            val.setStyleSheet("font-size: 12px;")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            details_layout.addWidget(lbl, row, 0, Qt.AlignRight)
            details_layout.addWidget(val, row, 1)
            self._detail_labels[key] = val

        layout.addWidget(details_group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 16, 16, 16)
        layout.setSpacing(12)

        # Settings row
        settings_group = QGroupBox("Analysis Settings")
        sg_layout = QGridLayout(settings_group)
        sg_layout.setSpacing(6)

        self._chk_ml = QCheckBox("AI Detection Models")
        self._chk_ml.setChecked(True)
        self._chk_freq = QCheckBox("Frequency Analysis (FFT)")
        self._chk_freq.setChecked(False)
        self._chk_noise = QCheckBox("Noise Pattern Analysis")
        self._chk_noise.setChecked(False)
        self._chk_calibration = QCheckBox("Social Media Calibration")
        self._chk_calibration.setChecked(True)

        sg_layout.addWidget(self._chk_ml, 0, 0)
        sg_layout.addWidget(self._chk_freq, 0, 1)
        sg_layout.addWidget(self._chk_noise, 1, 0)
        sg_layout.addWidget(self._chk_calibration, 1, 1)
        layout.addWidget(settings_group)

        # Analyze button + progress
        btn_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze Image")
        self._analyze_btn.setObjectName("accentBtn")
        self._analyze_btn.clicked.connect(self._run_analysis)
        self._analyze_btn.setEnabled(False)
        btn_row.addWidget(self._analyze_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setFixedWidth(180)
        self._progress_bar.hide()
        btn_row.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"font-size: 12px; color: {_COLORS['text_muted']};"
        )
        btn_row.addWidget(self._progress_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Verdict banner
        self._verdict_banner = VerdictBanner()
        layout.addWidget(self._verdict_banner)

        # Confidence bars
        conf_group = QGroupBox("Confidence Breakdown")
        conf_layout = QVBoxLayout(conf_group)

        ai_row = QHBoxLayout()
        ai_row.addWidget(QLabel("AI Probability:"))
        self._ai_bar = QProgressBar()
        self._ai_bar.setRange(0, 100)
        self._ai_bar.setValue(0)
        self._ai_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {_COLORS['danger']}; border-radius: 3px; }}"
        )
        ai_row.addWidget(self._ai_bar, 1)
        self._ai_pct = QLabel("- %")
        self._ai_pct.setFixedWidth(50)
        self._ai_pct.setAlignment(Qt.AlignRight)
        ai_row.addWidget(self._ai_pct)
        conf_layout.addLayout(ai_row)

        real_row = QHBoxLayout()
        real_row.addWidget(QLabel("Real Probability:"))
        self._real_bar = QProgressBar()
        self._real_bar.setRange(0, 100)
        self._real_bar.setValue(0)
        self._real_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {_COLORS['success']}; border-radius: 3px; }}"
        )
        real_row.addWidget(self._real_bar, 1)
        self._real_pct = QLabel("- %")
        self._real_pct.setFixedWidth(50)
        self._real_pct.setAlignment(Qt.AlignRight)
        real_row.addWidget(self._real_pct)
        conf_layout.addLayout(real_row)

        # Uncertainty row
        unc_row = QHBoxLayout()
        unc_row.addWidget(QLabel("Uncertainty:"))
        self._unc_bar = QProgressBar()
        self._unc_bar.setRange(0, 100)
        self._unc_bar.setValue(0)
        self._unc_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {_COLORS['warning']}; border-radius: 3px; }}"
        )
        unc_row.addWidget(self._unc_bar, 1)
        self._unc_pct = QLabel("- %")
        self._unc_pct.setFixedWidth(50)
        self._unc_pct.setAlignment(Qt.AlignRight)
        unc_row.addWidget(self._unc_pct)
        conf_layout.addLayout(unc_row)

        layout.addWidget(conf_group)

        # Calibrated thresholds info
        self._threshold_label = QLabel("")
        self._threshold_label.setStyleSheet(
            f"font-size: 11px; color: {_COLORS['text_muted']}; padding: 4px 8px;"
        )
        self._threshold_label.setWordWrap(True)
        layout.addWidget(self._threshold_label)

        # Calibrated CNN Ensemble section
        cnn_group = QGroupBox("Calibrated CNN Ensemble (3 Models + Temperature Scaling)")
        cnn_layout = QVBoxLayout(cnn_group)

        self._cnn_table = QTableWidget()
        self._cnn_table.setColumnCount(5)
        self._cnn_table.setHorizontalHeaderLabels(
            ["Model", "Raw P(AI)", "Temp (T)", "Calibrated P(AI)", "Verdict"]
        )
        self._cnn_table.horizontalHeader().setStretchLastSection(True)
        self._cnn_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._cnn_table.verticalHeader().setVisible(False)
        self._cnn_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._cnn_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._cnn_table.setAlternatingRowColors(True)
        self._cnn_table.setStyleSheet(
            f"alternate-background-color: {_COLORS['bg_secondary']};"
        )
        self._cnn_table.setMaximumHeight(160)
        cnn_layout.addWidget(self._cnn_table)

        self._ensemble_info = QLabel("")
        self._ensemble_info.setStyleSheet(
            f"font-size: 11px; color: {_COLORS['text_secondary']}; padding: 4px;"
        )
        self._ensemble_info.setWordWrap(True)
        cnn_layout.addWidget(self._ensemble_info)
        layout.addWidget(cnn_group)

        # Per-model results table (all methods)
        table_group = QGroupBox("All Detection Methods")
        tg_layout = QVBoxLayout(table_group)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(
            ["Method", "AI Prob", "Confidence", "Weight"]
        )
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setStyleSheet(
            f"alternate-background-color: {_COLORS['bg_secondary']};"
        )
        tg_layout.addWidget(self._results_table)
        layout.addWidget(table_group)

        # Forensics Analysis section (screenshot, platform, compression)
        forensics_group = QGroupBox("Forensic Analysis (Source & Provenance)")
        fg_layout = QVBoxLayout(forensics_group)

        self._forensics_table = QTableWidget()
        self._forensics_table.setColumnCount(4)
        self._forensics_table.setHorizontalHeaderLabels(
            ["Label", "Probability", "Confidence", "Evidence"]
        )
        self._forensics_table.horizontalHeader().setStretchLastSection(True)
        self._forensics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._forensics_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self._forensics_table.verticalHeader().setVisible(False)
        self._forensics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._forensics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._forensics_table.setAlternatingRowColors(True)
        self._forensics_table.setStyleSheet(
            f"alternate-background-color: {_COLORS['bg_secondary']};"
        )
        self._forensics_table.setMaximumHeight(180)
        fg_layout.addWidget(self._forensics_table)

        self._forensics_summary = QLabel("")
        self._forensics_summary.setStyleSheet(
            f"font-size: 11px; color: {_COLORS['text_secondary']}; padding: 4px;"
        )
        self._forensics_summary.setWordWrap(True)
        fg_layout.addWidget(self._forensics_summary)
        layout.addWidget(forensics_group)

        # Voting + timing
        meta_row = QHBoxLayout()
        self._votes_label = QLabel("")
        self._votes_label.setStyleSheet(
            f"font-size: 12px; color: {_COLORS['text_secondary']};"
        )
        meta_row.addWidget(self._votes_label)
        meta_row.addStretch()
        self._time_label = QLabel("")
        self._time_label.setStyleSheet(
            f"font-size: 12px; color: {_COLORS['text_muted']};"
        )
        meta_row.addWidget(self._time_label)
        layout.addLayout(meta_row)

        # Export buttons
        export_row = QHBoxLayout()
        self._export_json_btn = QPushButton("Export JSON Report")
        self._export_json_btn.clicked.connect(self._export_json)
        self._export_json_btn.setEnabled(False)
        export_row.addWidget(self._export_json_btn)

        self._new_scan_btn = QPushButton("New Scan")
        self._new_scan_btn.clicked.connect(self._reset)
        export_row.addWidget(self._new_scan_btn)
        export_row.addStretch()
        layout.addLayout(export_row)

        layout.addStretch()
        return panel

    def _setup_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet(
            f"background-color: {_COLORS['bg_secondary']}; "
            f"color: {_COLORS['text_secondary']}; "
            f"border-top: 1px solid {_COLORS['border']}; "
            f"font-size: 12px; padding: 2px 12px;"
        )

    # ------------------------------------------------------------------
    # Detector callback (from worker thread)
    # ------------------------------------------------------------------
    def _on_detector_ready(self, detector):
        """Called from worker thread when detector is first initialized."""
        self._detector = detector
        device = detector.device
        self._device_label.setText(f"Device: {device.upper()}")
        self.statusBar().showMessage(
            f"Models loaded on {device.upper()}  |  Analysis running..."
        )

    @staticmethod
    def _find_model() -> Optional[str]:
        candidates = [
            Path("models/best_model.pth"),
            Path("models/kaggle_deepfake.pth"),
            Path("models/swa_model.pth"),
        ]
        base = Path(__file__).resolve().parent.parent.parent.parent
        for c in candidates:
            if c.exists():
                return str(c)
            alt = base / c
            if alt.exists():
                return str(alt)
        return None

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tiff);;All Files (*)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            img = Image.open(io.BytesIO(raw))
            if img.mode != "RGB":
                img = img.convert("RGB")

            self._image = img
            self._image_bytes = raw
            self._image_path = path

            # Preview
            qimg = self._pil_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)
            self._drop_zone.set_image(pixmap)

            # Update details
            sz_kb = len(raw) / 1024.0
            ext = os.path.splitext(path)[1].lstrip(".").upper() or "?"
            bpp = len(raw) / (img.width * img.height) if img.width and img.height else 0

            self._detail_labels["file"].setText(os.path.basename(path))
            self._detail_labels["format"].setText(ext)
            self._detail_labels["dims"].setText(f"{img.width} x {img.height}")
            self._detail_labels["size"].setText(f"{sz_kb:,.1f} KB")
            self._detail_labels["bpp"].setText(f"{bpp:.3f}")
            self._detail_labels["sha256"].setText(
                hashlib.sha256(raw).hexdigest()[:20] + "..."
            )
            self._detail_labels["exif"].setText("-")
            self._detail_labels["compression"].setText("-")
            self._detail_labels["source"].setText("-")

            self._analyze_btn.setEnabled(True)
            self._verdict_banner.reset()
            self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")

        except Exception as exc:
            QMessageBox.warning(self, "Load Error", f"Cannot open image:\n{exc}")

    @staticmethod
    def _pil_to_qimage(pil_img: Image.Image) -> QImage:
        data = pil_img.tobytes("raw", "RGB")
        return QImage(
            data, pil_img.width, pil_img.height,
            pil_img.width * 3, QImage.Format_RGB888
        ).copy()  # .copy() detaches from Python memory

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self._image is None:
            return

        self._analyze_btn.setEnabled(False)
        self._export_json_btn.setEnabled(False)
        self._progress_bar.show()
        self._progress_label.setText(
            "Loading models..." if self._detector is None else "Analyzing..."
        )
        self._verdict_banner.reset()
        self.statusBar().showMessage(
            "Loading AI models (first run)..."
            if self._detector is None
            else "Analysis in progress..."
        )

        settings = {
            "use_ml": self._chk_ml.isChecked(),
            "use_frequency": self._chk_freq.isChecked(),
            "use_noise": self._chk_noise.isChecked(),
            "auto_calibration": self._chk_calibration.isChecked(),
        }

        self._worker = AnalysisWorker(
            self._detector, self._image, self._image_bytes,
            self._image_path, settings,
            kaggle_path=self._find_model(),
        )
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.detector_ready.connect(self._on_detector_ready)
        self._worker.progress.connect(
            lambda msg: self._progress_label.setText(msg)
        )
        self._worker.start()

    def _on_analysis_done(self, result: dict):
        self._result = result
        self._progress_bar.hide()
        self._progress_label.setText("")
        self._analyze_btn.setEnabled(True)
        self._export_json_btn.setEnabled(True)

        # Combined verdict: multi-signal fusion
        # Sources: CNN ensemble, HuggingFace models, forensics authenticity
        cal_ens = result.get("calibrated_ensemble")
        score = result.get("score_info", {})
        elapsed = result.get("elapsed_s", 0)
        indiv = result.get("individual_results", [])
        forensics = result.get("forensics")

        # Extract HuggingFace model probabilities (not CNN, not signal)
        hf_entries = [
            r for r in indiv
            if r.get("method", "").startswith("ML:")
            and "calibrated" not in r.get("method", "").lower()
            and "Ensemble" not in r.get("method", "")
            and "Custom Trained" not in r.get("method", "")
        ]
        hf_probs = [r["ai_probability"] for r in hf_entries]
        hf_ai_avg = sum(hf_probs) / len(hf_probs) if hf_probs else 0.5
        hf_ai_max = max(hf_probs) if hf_probs else 0.5
        hf_ai_votes = sum(1 for p in hf_probs if p > 0.5)

        # Debug: log per-model outputs
        for r in hf_entries:
            print(f"  [HF] {r.get('method', '?')}: P(AI)={r['ai_probability']:.4f}")
        print(f"  [HF] avg={hf_ai_avg:.4f}  max={hf_ai_max:.4f}  votes={hf_ai_votes}/{len(hf_probs)}")

        # Forensics authenticity signal (0=definitely not authentic, 1=authentic)
        forensics_authenticity = 1.0
        if forensics and isinstance(forensics, dict):
            forensics_authenticity = forensics.get("authenticity_score", 1.0)
            print(f"  [Forensics] authenticity={forensics_authenticity:.4f}  "
                  f"label={forensics.get('primary_label', '?')}")

        if cal_ens and cal_ens.get("calibrated_probs"):
            if cal_ens["strategy"] == "min":
                cnn_prob = cal_ens["ensemble_min_prob"]
            else:
                cnn_prob = cal_ens["ensemble_avg_prob"]
            low_t = cal_ens["uncertain_low"]
            high_t = cal_ens["uncertain_high"]
            print(f"  [CNN] prob={cnn_prob:.4f}  thresholds=[{low_t:.2f}, {high_t:.2f}]")

            # Multi-signal fusion:
            # 1. max(CNN, HF_avg) as base
            # 2. If any single HF model is very confident (>0.7), use that
            # 3. If forensics says low authenticity, boost the AI signal
            ai_prob = max(cnn_prob, hf_ai_avg, hf_ai_max * 0.9)

            # Forensics boost: if authenticity is very low, it's suspicious
            if forensics_authenticity < 0.3:
                # Low authenticity = likely not a real camera photo
                # Boost AI probability: blend with (1 - authenticity)
                forensic_ai_signal = 1.0 - forensics_authenticity
                ai_prob = max(ai_prob, ai_prob * 0.5 + forensic_ai_signal * 0.5)
                print(f"  [Fusion] forensics boost: ai_prob -> {ai_prob:.4f}")

            print(f"  [Fusion] final ai_prob={ai_prob:.4f}")

            # Three-way classification on combined probability
            if ai_prob >= high_t:
                verdict = "ai_generated"
            elif ai_prob < low_t:
                verdict = "real"
            else:
                verdict = "uncertain"

            # Safety: if CNN and HF strongly disagree, mark uncertain
            # But NOT if forensics also flags it
            if abs(cnn_prob - hf_ai_avg) > 0.50 and forensics_authenticity > 0.5:
                verdict = "uncertain"
        else:
            # Fallback: no CNN ensemble, use HF + score_info
            cnn_prob = None
            ai_prob = score.get("ai_prob", result.get("ai_probability", 0.5))
            verdict = self._calibrated_verdict(ai_prob, score)
            low_t, high_t = 0.34, 0.54

        real_prob = 1.0 - ai_prob

        subtitle = f"Analysis completed in {elapsed:.1f}s"
        if hf_probs and cal_ens:
            subtitle += (f"  |  CNN={cnn_prob:.0%} HF-avg={hf_ai_avg:.0%}"
                         f" HF-max={hf_ai_max:.0%}")
        if forensics_authenticity < 0.5:
            subtitle += f"  |  Authenticity={forensics_authenticity:.0%}"
        if result.get("uncertainty"):
            unc = result["uncertainty"]
            if unc.get("should_abstain"):
                subtitle += "  |  High uncertainty detected"

        self._verdict_banner.set_verdict(verdict, ai_prob, subtitle)

        # Confidence bars
        self._ai_bar.setValue(int(ai_prob * 100))
        self._ai_pct.setText(f"{ai_prob:.1%}")
        self._real_bar.setValue(int(real_prob * 100))
        self._real_pct.setText(f"{real_prob:.1%}")

        unc_score = 0.0
        if result.get("uncertainty"):
            unc_score = result["uncertainty"].get("score", 0.0)
        self._unc_bar.setValue(int(unc_score * 100))
        self._unc_pct.setText(f"{unc_score:.1%}")

        # Threshold info
        self._threshold_label.setText(
            f"Calibrated on 160,705 validation samples (1,000 bootstrap)  |  "
            f"UNCERTAIN region: [{low_t:.2f}, {high_t:.2f}]  |  "
            f"Strategy: {cal_ens.get('strategy', 'min') if cal_ens else 'default'}  |  "
            f"Confident accuracy: 99.5%"
        )

        # CNN Ensemble table
        self._cnn_table.setRowCount(0)
        if cal_ens and cal_ens.get("calibrated_probs"):
            raw = cal_ens.get("raw_probs", {})
            cal = cal_ens["calibrated_probs"]
            models = list(cal.keys())
            self._cnn_table.setRowCount(len(models) + 1)  # +1 for ensemble row

            for i, model_name in enumerate(models):
                raw_p = raw.get(model_name, 0)
                cal_p = cal.get(model_name, 0)

                # Find temperature from individual results details
                temp_val = 1.0
                for ir in result.get("individual_results", []):
                    if model_name in ir.get("method", ""):
                        temp_val = ir.get("details", {}).get("temperature", 1.0)
                        break

                self._cnn_table.setItem(i, 0, QTableWidgetItem(model_name))

                raw_item = QTableWidgetItem(f"{raw_p:.4f}")
                raw_item.setTextAlignment(Qt.AlignCenter)
                self._cnn_table.setItem(i, 1, raw_item)

                temp_item = QTableWidgetItem(f"{temp_val:.4f}")
                temp_item.setTextAlignment(Qt.AlignCenter)
                temp_item.setForeground(QColor(_COLORS["info"]))
                self._cnn_table.setItem(i, 2, temp_item)

                cal_item = QTableWidgetItem(f"{cal_p:.4f}")
                cal_item.setTextAlignment(Qt.AlignCenter)
                if cal_p >= high_t:
                    cal_item.setForeground(QColor(_COLORS["danger"]))
                elif cal_p < low_t:
                    cal_item.setForeground(QColor(_COLORS["success"]))
                else:
                    cal_item.setForeground(QColor(_COLORS["warning"]))
                self._cnn_table.setItem(i, 3, cal_item)

                # Per-model verdict
                if cal_p >= 0.54:
                    v_text = "AI"
                elif cal_p < 0.34:
                    v_text = "REAL"
                else:
                    v_text = "UNCERTAIN"
                v_item = QTableWidgetItem(v_text)
                v_item.setTextAlignment(Qt.AlignCenter)
                v_item.setForeground(QColor(
                    _COLORS["danger"] if v_text == "AI"
                    else _COLORS["success"] if v_text == "REAL"
                    else _COLORS["warning"]
                ))
                self._cnn_table.setItem(i, 4, v_item)

            # Ensemble row
            ens_row = len(models)
            strategy = cal_ens.get("strategy", "min")
            ens_p = cal_ens["ensemble_min_prob"] if strategy == "min" else cal_ens["ensemble_avg_prob"]

            ens_name = QTableWidgetItem(f"ENSEMBLE ({strategy.upper()})")
            ens_name.setFont(QFont("Segoe UI", 11, QFont.Bold))
            self._cnn_table.setItem(ens_row, 0, ens_name)

            self._cnn_table.setItem(ens_row, 1, QTableWidgetItem(""))

            self._cnn_table.setItem(ens_row, 2, QTableWidgetItem(""))

            ens_cal = QTableWidgetItem(f"{ens_p:.4f}")
            ens_cal.setTextAlignment(Qt.AlignCenter)
            ens_cal.setFont(QFont("Segoe UI", 11, QFont.Bold))
            ens_cal.setForeground(QColor(
                _COLORS["danger"] if verdict == "ai_generated"
                else _COLORS["success"] if verdict == "real"
                else _COLORS["warning"]
            ))
            self._cnn_table.setItem(ens_row, 3, ens_cal)

            ens_verdict = QTableWidgetItem(verdict.upper().replace("_", " "))
            ens_verdict.setTextAlignment(Qt.AlignCenter)
            ens_verdict.setFont(QFont("Segoe UI", 11, QFont.Bold))
            ens_verdict.setForeground(QColor(
                _COLORS["danger"] if verdict == "ai_generated"
                else _COLORS["success"] if verdict == "real"
                else _COLORS["warning"]
            ))
            self._cnn_table.setItem(ens_row, 4, ens_verdict)

            self._cnn_table.resizeRowsToContents()

            # Ensemble info label
            std = cal_ens.get("ensemble_std", 0)
            agreement = cal_ens.get("model_agreement", 0)
            self._ensemble_info.setText(
                f"Ensemble std: {std:.4f}  |  "
                f"Model agreement: {agreement:.0%}  |  "
                f"Avg: {cal_ens['ensemble_avg_prob']:.4f}  |  "
                f"Min: {cal_ens['ensemble_min_prob']:.4f}"
            )
        else:
            self._ensemble_info.setText(
                "CNN ensemble not available (model weights not found)"
            )

        # All methods results table
        indiv = result.get("individual_results", [])
        self._results_table.setRowCount(len(indiv))
        for i, r in enumerate(indiv):
            method = r.get("method", "?")
            prob = r.get("ai_probability", 0)
            conf = r.get("confidence", 0)
            weight = r.get("weight", 0)

            self._results_table.setItem(i, 0, QTableWidgetItem(method))

            prob_item = QTableWidgetItem(f"{prob:.3f}")
            prob_item.setTextAlignment(Qt.AlignCenter)
            if prob > 0.54:
                prob_item.setForeground(QColor(_COLORS["danger"]))
            elif prob < 0.34:
                prob_item.setForeground(QColor(_COLORS["success"]))
            else:
                prob_item.setForeground(QColor(_COLORS["warning"]))
            self._results_table.setItem(i, 1, prob_item)

            conf_item = QTableWidgetItem(f"{conf:.2f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self._results_table.setItem(i, 2, conf_item)

            w_item = QTableWidgetItem(f"{weight:.2f}")
            w_item.setTextAlignment(Qt.AlignCenter)
            self._results_table.setItem(i, 3, w_item)

        self._results_table.resizeRowsToContents()

        # Forensics results table
        self._forensics_table.setRowCount(0)
        forensics = result.get("forensics")
        if forensics and forensics.get("labels"):
            labels = forensics["labels"]
            # Only show labels with probability > 0.1
            visible = [lb for lb in labels if lb.get("probability", 0) > 0.1]
            self._forensics_table.setRowCount(len(visible))
            for i, lb in enumerate(visible):
                label_name = lb.get("label", "?").replace("_", " ").title()
                prob_val = lb.get("probability", 0)
                conf_val = lb.get("confidence", "LOW")
                evidence_list = lb.get("evidence", [])

                name_item = QTableWidgetItem(label_name)
                name_item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                self._forensics_table.setItem(i, 0, name_item)

                prob_item = QTableWidgetItem(f"{prob_val:.0%}")
                prob_item.setTextAlignment(Qt.AlignCenter)
                if prob_val >= 0.7:
                    prob_item.setForeground(QColor(_COLORS["danger"]))
                elif prob_val >= 0.4:
                    prob_item.setForeground(QColor(_COLORS["warning"]))
                else:
                    prob_item.setForeground(QColor(_COLORS["text_muted"]))
                self._forensics_table.setItem(i, 1, prob_item)

                conf_item = QTableWidgetItem(str(conf_val))
                conf_item.setTextAlignment(Qt.AlignCenter)
                self._forensics_table.setItem(i, 2, conf_item)

                evidence_text = "; ".join(evidence_list[:3])
                if len(evidence_list) > 3:
                    evidence_text += f" (+{len(evidence_list) - 3} more)"
                self._forensics_table.setItem(
                    i, 3, QTableWidgetItem(evidence_text)
                )

            self._forensics_table.resizeRowsToContents()

            # Summary line
            summary_parts = []
            top_evidence = forensics.get("top_evidence", [])
            if top_evidence:
                summary_parts.append("Key findings: " + " | ".join(top_evidence[:3]))
            contradictions = forensics.get("contradictions", [])
            if contradictions:
                summary_parts.append("Contradictions: " + "; ".join(contradictions[:2]))
            self._forensics_summary.setText("  ".join(summary_parts))
        else:
            err = result.get("forensics_error", "")
            self._forensics_summary.setText(
                f"Forensics unavailable{': ' + err if err else ''}"
            )

        # Voting + timing
        votes = result.get("votes", {})
        self._votes_label.setText(
            f"Votes: {votes.get('ai', 0)} AI / {votes.get('real', 0)} Real "
            f"(of {votes.get('total', 0)})"
        )
        self._time_label.setText(f"Elapsed: {elapsed:.2f}s")

        # Update image details from analysis
        meta = result.get("metadata", {})
        src = result.get("source_info", {})
        self._detail_labels["exif"].setText("Yes" if meta.get("has_exif") else "No")
        self._detail_labels["compression"].setText(
            src.get("compression_level", "-").title()
        )
        # Platform: check forensics first, then source_info
        platform_text = src.get("platform") or "Local"
        if forensics and forensics.get("labels"):
            for lb in forensics["labels"]:
                if lb.get("label") == "social_media_likely" and lb.get("probability", 0) > 0.4:
                    platform_text = "Social Media (detected)"
                    break
                if lb.get("label") == "screenshot_likely" and lb.get("probability", 0) > 0.4:
                    platform_text = "Screenshot (detected)"
                    break
        self._detail_labels["source"].setText(platform_text)

        self.statusBar().showMessage(
            f"Analysis complete  |  {elapsed:.1f}s  |  "
            f"Verdict: {verdict.replace('_', ' ').title()}"
        )

    def _calibrated_verdict(self, ai_prob: float, score: dict) -> str:
        """Use empirically calibrated thresholds for verdict."""
        # Use UNCERTAIN region from calibrated_thresholds.json
        # Single model thresholds: [0.34, 0.54]
        low_t = 0.34
        high_t = 0.54

        calibration_applied = score.get("calibration_applied", False)
        if calibration_applied:
            # Social media: be more conservative
            low_t = 0.25
            high_t = 0.65

        uncertainty = self._result.get("uncertainty", {}) if self._result else {}
        if uncertainty.get("should_abstain"):
            return "uncertain"

        if ai_prob >= high_t:
            return "ai_generated"
        elif ai_prob < low_t:
            return "real"
        else:
            return "uncertain"

    def _on_analysis_error(self, msg: str):
        self._progress_bar.hide()
        self._progress_label.setText("")
        self._analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n\n{msg}")
        self.statusBar().showMessage("Analysis failed")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_json(self):
        if self._result is None:
            return

        default_name = "imagetrust_report.json"
        if self._image_path:
            stem = Path(self._image_path).stem
            default_name = f"imagetrust_{stem}.json"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON Report", default_name,
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return

        # Build serializable report
        cal_ens = self._result.get("calibrated_ensemble")
        report = {
            "tool": "ImageTrust Desktop v1.0",
            "report_time": self._result.get("report_time", ""),
            "image": {
                "path": self._image_path,
                "sha256": self._result.get("sha256", ""),
                "dimensions": (
                    f"{self._image.width}x{self._image.height}"
                    if self._image else ""
                ),
            },
            "verdict": self._result.get("verdict", ""),
            "ai_probability": self._result.get("ai_probability", 0),
            "real_probability": self._result.get("real_probability", 0),
            "confidence": self._result.get("confidence", 0),
            "uncertainty": self._result.get("uncertainty"),
            "votes": self._result.get("votes"),
            "elapsed_s": self._result.get("elapsed_s", 0),
            "individual_results": self._result.get("individual_results", []),
            "calibrated_cnn_ensemble": {
                "raw_probs": cal_ens.get("raw_probs", {}) if cal_ens else {},
                "calibrated_probs": cal_ens.get("calibrated_probs", {}) if cal_ens else {},
                "ensemble_avg_prob": cal_ens.get("ensemble_avg_prob", 0) if cal_ens else None,
                "ensemble_min_prob": cal_ens.get("ensemble_min_prob", 0) if cal_ens else None,
                "verdict": cal_ens.get("verdict", "") if cal_ens else "",
                "strategy": cal_ens.get("strategy", "") if cal_ens else "",
                "ensemble_std": cal_ens.get("ensemble_std", 0) if cal_ens else None,
                "model_agreement": cal_ens.get("model_agreement", 0) if cal_ens else None,
            },
            "calibration": {
                "dataset_size": 160705,
                "bootstrap_iterations": 1000,
                "uncertain_region_single": {"low": 0.34, "high": 0.54},
                "uncertain_region_ensemble_min": {"low": 0.52, "high": 0.57},
                "confident_accuracy": 0.995,
                "temperature_scaling": {
                    "ResNet-50": 0.6401,
                    "EfficientNetV2-M": 0.5641,
                    "ConvNeXt-Base": 0.6457,
                },
            },
            "forensics": self._result.get("forensics"),
            "forensics_details": self._result.get("forensics_results", []),
            "source_info": self._result.get("source_info"),
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            self.statusBar().showMessage(f"Report exported: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Error", f"Failed to save:\n{exc}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset(self):
        self._image = None
        self._image_bytes = None
        self._image_path = None
        self._result = None

        self._drop_zone._set_placeholder()
        self._analyze_btn.setEnabled(False)
        self._export_json_btn.setEnabled(False)
        self._verdict_banner.reset()
        self._ai_bar.setValue(0)
        self._ai_pct.setText("- %")
        self._real_bar.setValue(0)
        self._real_pct.setText("- %")
        self._unc_bar.setValue(0)
        self._unc_pct.setText("- %")
        self._cnn_table.setRowCount(0)
        self._ensemble_info.setText("")
        self._results_table.setRowCount(0)
        self._forensics_table.setRowCount(0)
        self._forensics_summary.setText("")
        self._votes_label.setText("")
        self._time_label.setText("")
        self._threshold_label.setText("")
        for v in self._detail_labels.values():
            v.setText("-")
        self.statusBar().showMessage("Ready  |  Load an image to begin")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLESHEET)

    # Dark palette (backup for elements not covered by stylesheet)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(_COLORS["bg_primary"]))
    palette.setColor(QPalette.WindowText, QColor(_COLORS["text_primary"]))
    palette.setColor(QPalette.Base, QColor(_COLORS["bg_card"]))
    palette.setColor(QPalette.AlternateBase, QColor(_COLORS["bg_secondary"]))
    palette.setColor(QPalette.Text, QColor(_COLORS["text_primary"]))
    palette.setColor(QPalette.Button, QColor(_COLORS["bg_input"]))
    palette.setColor(QPalette.ButtonText, QColor(_COLORS["text_primary"]))
    palette.setColor(QPalette.Highlight, QColor(_COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    window = ImageTrustWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    # Ensure src/ is on path when run directly
    _src = str(Path(__file__).resolve().parent.parent.parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    main()
