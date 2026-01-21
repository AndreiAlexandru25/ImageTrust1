"""
Forensic report generator.
"""

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

from imagetrust.core.types import AnalysisResult
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


class ForensicReportGenerator:
    """
    Generates comprehensive forensic reports.
    
    Supports multiple output formats:
    - PDF: Professional document with visuals
    - HTML: Interactive web report
    - JSON: Machine-readable data
    
    Example:
        >>> generator = ForensicReportGenerator()
        >>> report_path = generator.generate(analysis_result, "pdf")
    """

    def __init__(
        self,
        output_dir: Optional[Union[Path, str]] = None,
        template_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.template_dir = Path(template_dir) if template_dir else None
        
        ensure_dir(self.output_dir)

    def generate(
        self,
        analysis: AnalysisResult,
        format: str = "pdf",
        filename: Optional[str] = None,
        include_explainability: bool = True,
        include_metadata: bool = True,
    ) -> Path:
        """
        Generate a forensic report.
        
        Args:
            analysis: AnalysisResult from image analysis
            format: Output format ("pdf", "html", "json")
            filename: Optional custom filename
            include_explainability: Include Grad-CAM visuals
            include_metadata: Include metadata analysis
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{analysis.analysis_id}_{timestamp}"
        
        if format == "pdf":
            return self._generate_pdf(analysis, filename, include_explainability, include_metadata)
        elif format == "html":
            return self._generate_html(analysis, filename, include_explainability, include_metadata)
        elif format == "json":
            return self._generate_json(analysis, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_pdf(
        self,
        analysis: AnalysisResult,
        filename: str,
        include_explainability: bool,
        include_metadata: bool,
    ) -> Path:
        """Generate PDF report."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
        except ImportError:
            logger.error("reportlab not installed. Cannot generate PDF.")
            return self._generate_json(analysis, filename)
        
        output_path = self.output_dir / f"{filename}.pdf"
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
        )
        story.append(Paragraph("ImageTrust Forensic Report", title_style))
        
        # Analysis info
        story.append(Paragraph(f"<b>Analysis ID:</b> {analysis.analysis_id}", styles["Normal"]))
        story.append(Paragraph(f"<b>Timestamp:</b> {analysis.timestamp.isoformat()}", styles["Normal"]))
        story.append(Spacer(1, 20))
        
        # Main result
        story.append(Paragraph("Detection Result", styles["Heading2"]))
        
        result_data = [
            ["Metric", "Value"],
            ["Verdict", analysis.verdict.value.replace("_", " ").title()],
            ["AI Probability", f"{analysis.ai_probability:.1%}"],
            ["Confidence", analysis.confidence.value.replace("_", " ").title()],
            ["Processing Time", f"{analysis.processing_time_ms:.0f} ms"],
        ]
        
        result_table = Table(result_data, colWidths=[2*inch, 3*inch])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#CCCCCC")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 20))
        
        # Models used
        if analysis.models_used:
            story.append(Paragraph("Models Used", styles["Heading2"]))
            for model in analysis.models_used:
                story.append(Paragraph(f"• {model}", styles["Normal"]))
            story.append(Spacer(1, 20))
        
        # Metadata
        if include_metadata and analysis.metadata:
            story.append(Paragraph("Metadata Analysis", styles["Heading2"]))
            meta = analysis.metadata
            story.append(Paragraph(f"<b>Has Metadata:</b> {'Yes' if meta.has_metadata else 'No'}", styles["Normal"]))
            if meta.exif:
                story.append(Paragraph(f"<b>Camera:</b> {meta.exif.make or 'Unknown'} {meta.exif.model or ''}", styles["Normal"]))
            if meta.ai_indicators:
                story.append(Paragraph("<b>AI Indicators:</b>", styles["Normal"]))
                for indicator in meta.ai_indicators:
                    story.append(Paragraph(f"• {indicator}", styles["Normal"]))
            story.append(Spacer(1, 20))
        
        # Warnings
        if analysis.warnings:
            story.append(Paragraph("Warnings", styles["Heading2"]))
            for warning in analysis.warnings:
                story.append(Paragraph(f"⚠️ {warning}", styles["Normal"]))
            story.append(Spacer(1, 20))
        
        # Summary
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(Paragraph(analysis.get_summary(), styles["Normal"]))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"<i>Generated by ImageTrust v0.1.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            styles["Normal"]
        ))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def _generate_html(
        self,
        analysis: AnalysisResult,
        filename: str,
        include_explainability: bool,
        include_metadata: bool,
    ) -> Path:
        """Generate HTML report."""
        output_path = self.output_dir / f"{filename}.html"
        
        # Determine verdict color
        if analysis.verdict.value == "ai_generated":
            verdict_color = "#dc3545"
            verdict_bg = "#fff5f5"
        elif analysis.verdict.value == "real":
            verdict_color = "#28a745"
            verdict_bg = "#f0fff4"
        else:
            verdict_color = "#ffc107"
            verdict_bg = "#fffbeb"
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImageTrust Forensic Report - {analysis.analysis_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #1E3A5F, #2D5A87); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 2rem; margin-bottom: 10px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #1E3A5F; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #eee; }}
        .result-box {{ background: {verdict_bg}; border: 2px solid {verdict_color}; border-radius: 10px; padding: 20px; text-align: center; }}
        .result-box .verdict {{ font-size: 1.5rem; font-weight: bold; color: {verdict_color}; }}
        .result-box .probability {{ font-size: 2rem; font-weight: bold; margin: 10px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric .value {{ font-size: 1.5rem; font-weight: bold; color: #1E3A5F; }}
        .metric .label {{ font-size: 0.9rem; color: #666; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 10px 0; border-radius: 0 5px 5px 0; }}
        .footer {{ text-align: center; color: #666; font-size: 0.9rem; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 ImageTrust Forensic Report</h1>
            <p>Analysis ID: {analysis.analysis_id}</p>
            <p>Generated: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="card">
            <h2>Detection Result</h2>
            <div class="result-box">
                <div class="verdict">{analysis.verdict.value.replace('_', ' ').upper()}</div>
                <div class="probability">{analysis.ai_probability:.1%} AI</div>
                <div>Confidence: {analysis.confidence.value.replace('_', ' ').title()}</div>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="value">{analysis.ai_probability:.1%}</div>
                    <div class="label">AI Probability</div>
                </div>
                <div class="metric">
                    <div class="value">{1 - analysis.ai_probability:.1%}</div>
                    <div class="label">Real Probability</div>
                </div>
                <div class="metric">
                    <div class="value">{analysis.processing_time_ms:.0f}ms</div>
                    <div class="label">Processing Time</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Summary</h2>
            <p>{analysis.get_summary()}</p>
        </div>
        
        <div class="footer">
            <p>Generated by ImageTrust v0.1.0</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _generate_json(
        self,
        analysis: AnalysisResult,
        filename: str,
    ) -> Path:
        """Generate JSON report."""
        output_path = self.output_dir / f"{filename}.json"
        
        report_data = analysis.to_report_dict()
        report_data["generated_at"] = datetime.now().isoformat()
        report_data["generator"] = "ImageTrust v0.1.0"
        
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {output_path}")
        return output_path
