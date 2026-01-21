"""
ImageTrust Desktop App (Tkinter)
Local, offline-friendly AI image analysis for Windows EXE packaging.
"""

import io
import os
import hashlib
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from types import SimpleNamespace

from PIL import Image, ImageTk

from imagetrust.detection.multi_detector import ComprehensiveDetector
from imagetrust.utils.scoring import analyze_image_source, compute_combined_score


def extract_metadata_minimal(image_bytes: bytes) -> dict:
    """Minimal EXIF check used for calibration."""
    metadata = {"has_exif": False}
    try:
        import exifread
        from io import BytesIO
        tags = exifread.process_file(BytesIO(image_bytes), details=False)
        metadata["has_exif"] = bool(tags)
    except Exception:
        metadata["has_exif"] = False
    return metadata


class ImageTrustDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ImageTrust Desktop")
        self.root.geometry("1280x820")
        self.root.configure(bg="#111318")

        self.detector = ComprehensiveDetector()

        self.image = None
        self.image_bytes = None
        self.image_path = None
        self.image_preview = None
        self.info_labels = {}

        self._build_ui()

    def _build_ui(self):
        # Layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(16, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(header, text="ImageTrust Desktop", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Offline AI Image Forensics", style="Subtitle.TLabel").pack(anchor="w")

        content = ttk.Frame(self.root, padding=12)
        content.grid(row=1, column=0, columnspan=2, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(content, padding=12)
        right = ttk.Frame(content, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")

        # Left panel (image)
        ttk.Label(left, text="Upload Image", style="Section.TLabel").pack(anchor="w")
        ttk.Button(left, text="Choose Image", command=self.load_image, style="Accent.TButton").pack(anchor="w", pady=6)

        self.path_label = ttk.Label(left, text="No file selected", foreground="#888888")
        self.path_label.pack(anchor="w", pady=(0, 8))

        self.image_label = ttk.Label(left)
        self.image_label.pack(fill="both", expand=True)

        # Right panel (results)
        ttk.Label(right, text="Forensic Report", style="Section.TLabel").pack(anchor="w")

        settings_frame = ttk.LabelFrame(right, text="Analysis Settings", padding=8, style="Section.TLabelframe")
        settings_frame.pack(fill="x", pady=8)

        self.use_ml = tk.BooleanVar(value=True)
        self.use_frequency = tk.BooleanVar(value=False)
        self.use_noise = tk.BooleanVar(value=False)
        self.auto_calibration = tk.BooleanVar(value=True)
        self.high_confidence = tk.BooleanVar(value=True)
        self.show_details = tk.BooleanVar(value=True)

        ttk.Checkbutton(settings_frame, text="AI Detection Models", variable=self.use_ml).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="Frequency Analysis", variable=self.use_frequency).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="Noise Pattern Analysis", variable=self.use_noise).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="Auto Calibration (WhatsApp/Instagram)", variable=self.auto_calibration).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="High Confidence (>=75%)", variable=self.high_confidence).grid(row=4, column=0, sticky="w")
        ttk.Checkbutton(settings_frame, text="Show Detailed Report", variable=self.show_details).grid(row=5, column=0, sticky="w")

        ttk.Button(right, text="Run Analysis", command=self.run_analysis, style="Accent.TButton").pack(anchor="w", pady=8)

        # Summary box
        summary = ttk.LabelFrame(right, text="Summary", padding=8, style="Section.TLabelframe")
        summary.pack(fill="x", pady=(6, 8))

        self.verdict_label = ttk.Label(summary, text="Verdict: -", style="Verdict.TLabel")
        self.verdict_label.pack(anchor="w")

        self.prob_label = ttk.Label(summary, text="AI Probability: - | Real Probability: -", style="Body.TLabel")
        self.prob_label.pack(anchor="w", pady=(4, 2))

        self.conf_label = ttk.Label(summary, text="Confidence: -", style="Body.TLabel")
        self.conf_label.pack(anchor="w")

        self.ai_bar = ttk.Progressbar(summary, orient="horizontal", length=360, mode="determinate")
        self.ai_bar.pack(anchor="w", pady=(6, 0))

        self.note_label = ttk.Label(summary, text="", style="Hint.TLabel", wraplength=520)
        self.note_label.pack(anchor="w", pady=(4, 2))

        # Image details
        details_frame = ttk.LabelFrame(right, text="Image Details", padding=8, style="Section.TLabelframe")
        details_frame.pack(fill="x", pady=(0, 8))

        info_keys = [
            ("File", "file"),
            ("Format", "format"),
            ("Dimensions", "dimensions"),
            ("File Size", "size"),
            ("EXIF", "exif"),
            ("Compression", "compression"),
            ("Source", "source"),
            ("BPP", "bpp"),
            ("SHA256", "sha256"),
            ("Report Time", "report_time"),
        ]
        for i, (label, key) in enumerate(info_keys):
            ttk.Label(details_frame, text=f"{label}:", style="Body.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 8))
            value_label = ttk.Label(details_frame, text="-", style="Body.TLabel")
            value_label.grid(row=i, column=1, sticky="w")
            self.info_labels[key] = value_label

        # Detailed report
        report_frame = ttk.LabelFrame(right, text="Detailed Report", padding=8, style="Section.TLabelframe")
        report_frame.pack(fill="both", expand=True)

        self.details_box = scrolledtext.ScrolledText(report_frame, height=18, wrap="word")
        self.details_box.pack(fill="both", expand=True)
        self.details_box.insert("end", "Detailed analysis will appear here.\n")
        self.details_box.configure(state="disabled", font=("Consolas", 10), background="#0F1117", foreground="#E5E7EB", insertbackground="#E5E7EB")

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
            image = Image.open(io.BytesIO(image_bytes))

            if image.mode != "RGB":
                image = image.convert("RGB")

            self.image = image
            self.image_bytes = image_bytes
            self.image_path = path
            self.path_label.configure(text=path)

            # Preview
            preview = image.copy()
            preview.thumbnail((480, 480))
            self.image_preview = ImageTk.PhotoImage(preview)
            self.image_label.configure(image=self.image_preview)

            # Update image details
            file_size_kb = len(image_bytes) / 1024.0
            file_ext = os.path.splitext(path)[1].lstrip(".").upper() or "Unknown"
            self.info_labels["file"].configure(text=os.path.basename(path))
            self.info_labels["format"].configure(text=file_ext)
            self.info_labels["dimensions"].configure(text=f"{image.width} × {image.height}")
            self.info_labels["size"].configure(text=f"{file_size_kb:.1f} KB")
            self.info_labels["exif"].configure(text="Unknown")
            self.info_labels["compression"].configure(text="Unknown")
            self.info_labels["source"].configure(text="Local")
            self.info_labels["bpp"].configure(text="—")
            self.info_labels["sha256"].configure(text="—")
            self.info_labels["report_time"].configure(text="—")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def run_analysis(self):
        if self.image is None or self.image_bytes is None:
            messagebox.showwarning("No image", "Please select an image first.")
            return

        try:
            # Analyze with core detector
            result = self.detector.analyze(self.image)

            # Build scoring inputs
            metadata = extract_metadata_minimal(self.image_bytes)
            uploaded_file = SimpleNamespace(name=self.image_path or "image.jpg")
            source_info = analyze_image_source(self.image, self.image_bytes, uploaded_file, metadata)

            settings = {
                "use_ml": self.use_ml.get(),
                "use_frequency": self.use_frequency.get(),
                "use_noise": self.use_noise.get(),
                "auto_calibration": self.auto_calibration.get(),
            }

            score_info = compute_combined_score(
                result,
                uploaded_file,
                source_info=source_info,
                settings=settings,
            )

            combined = score_info["combined"]
            ai_prob = score_info["ai_prob"]
            real_prob = score_info["real_prob"]

            if self.high_confidence.get():
                real_threshold = 0.25
                ai_threshold = 0.75
            else:
                real_threshold = 0.35
                ai_threshold = 0.60

            if combined >= ai_threshold:
                verdict = "AI Generated"
                verdict_color = "#EF4444"
            elif combined <= real_threshold:
                verdict = "Real Photograph"
                verdict_color = "#22C55E"
            else:
                verdict = "Inconclusive"
                verdict_color = "#F59E0B"

            self.verdict_label.configure(text=f"Verdict: {verdict}", foreground=verdict_color)
            self.prob_label.configure(text=f"AI Probability: {ai_prob:.1%} | Real Probability: {real_prob:.1%}")

            confidence = max(ai_prob, real_prob)
            self.conf_label.configure(text=f"Confidence: {confidence:.1%}")
            self.ai_bar["value"] = ai_prob * 100

            note_parts = []
            if score_info.get("calibration_applied"):
                note_parts.append(score_info.get("calibration_note", "Auto calibration applied."))
            if score_info.get("signal_suppressed"):
                note_parts.append("Signal analysis suppressed due to compression.")
            self.note_label.configure(text=" ".join(note_parts))

            # Details
            self.details_box.configure(state="normal")
            self.details_box.delete("1.0", "end")
            self.details_box.insert("end", "Active Methods:\n")
            for r in score_info.get("active_results", []):
                self.details_box.insert("end", f"- {r['method']}: {r['ai_probability']:.1%} AI\n")
            self.details_box.insert("end", "\nWeights:\n")
            self.details_box.insert("end", f"ML: {score_info['weights']['ml']} | Signal: {score_info['weights']['signal']} | Vote: {score_info['weights']['vote']}\n")
            self.details_box.insert("end", "\nReport Metadata:\n")
            hash_hex = hashlib.sha256(self.image_bytes).hexdigest()
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.details_box.insert("end", f"- SHA256: {hash_hex}\n")
            self.details_box.insert("end", f"- Report Time: {report_time}\n")

            if not self.show_details.get():
                self.details_box.insert("end", "\n(Enable 'Show Detailed Report' for more diagnostics.)\n")
            self.details_box.configure(state="disabled")

            # Update image details from analysis
            file_size_kb = len(self.image_bytes) / 1024.0
            ext = os.path.splitext(self.image_path or "")[1].lstrip(".").upper() or "Unknown"
            self.info_labels["file"].configure(text=os.path.basename(self.image_path or "image"))
            self.info_labels["format"].configure(text=ext)
            self.info_labels["dimensions"].configure(text=f"{self.image.width} × {self.image.height}")
            self.info_labels["size"].configure(text=f"{file_size_kb:.1f} KB")
            self.info_labels["exif"].configure(text="Yes" if metadata.get("has_exif") else "No")
            self.info_labels["compression"].configure(text=source_info.get("compression_level", "unknown").title())
            self.info_labels["source"].configure(text=(source_info.get("platform") or "Local"))
            self.info_labels["bpp"].configure(text=f"{source_info.get('bpp', 0):.2f}")
            self.info_labels["sha256"].configure(text=f"{hash_hex[:12]}…")
            self.info_labels["report_time"].configure(text=report_time)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")


def main():
    root = tk.Tk()
    # Basic ttk styling
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#111318")
    style.configure("TLabel", background="#111318", foreground="#E5E7EB", font=("Segoe UI", 10))
    style.configure("Body.TLabel", background="#111318", foreground="#E5E7EB", font=("Segoe UI", 10))
    style.configure("Title.TLabel", background="#111318", foreground="#F9FAFB", font=("Segoe UI", 18, "bold"))
    style.configure("Subtitle.TLabel", background="#111318", foreground="#9CA3AF", font=("Segoe UI", 10))
    style.configure("Section.TLabel", background="#111318", foreground="#F3F4F6", font=("Segoe UI", 13, "bold"))
    style.configure("Verdict.TLabel", background="#111318", foreground="#E5E7EB", font=("Segoe UI", 12, "bold"))
    style.configure("Hint.TLabel", background="#111318", foreground="#9CA3AF", font=("Segoe UI", 9))
    style.configure("TButton", background="#2a2f3a", foreground="#E5E7EB", font=("Segoe UI", 10, "bold"))
    style.configure("Accent.TButton", background="#4F46E5", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("active", "#4338CA")])
    style.configure("TCheckbutton", background="#111318", foreground="#E5E7EB", font=("Segoe UI", 10))
    style.configure("TLabelframe", background="#111318", foreground="#E5E7EB")
    style.configure("Section.TLabelframe", background="#111318", foreground="#E5E7EB")
    style.configure("TLabelframe.Label", background="#111318", foreground="#E5E7EB", font=("Segoe UI", 10, "bold"))

    app = ImageTrustDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
