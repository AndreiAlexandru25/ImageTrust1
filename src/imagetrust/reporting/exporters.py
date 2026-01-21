"""
Report exporters for different formats.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import io
import base64

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


class JSONExporter:
    """Export reports to JSON format."""
    
    def export(
        self,
        report: dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """Export report to JSON."""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return output_path


class PDFExporter:
    """Export reports to PDF format."""
    
    def __init__(self) -> None:
        """Initialize PDF exporter."""
        self.page_width = 595  # A4 width in points
        self.page_height = 842  # A4 height in points
        self.margin = 50
    
    def export(
        self,
        report: dict[str, Any],
        output_path: Path | str,
        include_images: bool = True,
    ) -> Path:
        """
        Export report to PDF.
        
        Args:
            report: Report dictionary
            output_path: Output file path
            include_images: Include visualization images
        
        Returns:
            Path to saved file
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image as RLImage, PageBreak
            )
            from reportlab.lib.units import inch
        except ImportError:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
        
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#1a1a2e'),
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=HexColor('#16213e'),
        )
        
        # Build content
        content = []
        
        # Title
        content.append(Paragraph("ImageTrust Forensic Report", title_style))
        content.append(Spacer(1, 20))
        
        # Report metadata
        content.append(Paragraph(f"Report ID: {report.get('report_id', 'N/A')}", styles['Normal']))
        content.append(Paragraph(f"Generated: {report.get('generated_at', 'N/A')}", styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Summary section
        summary = report.get("summary", {})
        content.append(Paragraph("Executive Summary", heading_style))
        
        verdict = summary.get("verdict", "unknown").upper()
        verdict_color = "#dc3545" if verdict == "AI_GENERATED" else "#28a745" if verdict == "REAL" else "#ffc107"
        
        content.append(Paragraph(
            f"<b>Verdict:</b> <font color='{verdict_color}'>{verdict}</font>",
            styles['Normal']
        ))
        content.append(Paragraph(
            f"<b>AI Probability:</b> {summary.get('ai_probability', 'N/A')}",
            styles['Normal']
        ))
        content.append(Paragraph(
            f"<b>Confidence:</b> {summary.get('confidence_level', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 10))
        content.append(Paragraph(summary.get("verdict_description", ""), styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Detection Results
        detection = report.get("detection", {})
        if detection:
            content.append(Paragraph("Detection Analysis", heading_style))
            
            # Detection table
            detection_data = [
                ["Metric", "Value"],
                ["AI Probability", f"{detection.get('primary_result', {}).get('ai_probability', 0):.2%}"],
                ["Real Probability", f"{detection.get('primary_result', {}).get('real_probability', 0):.2%}"],
                ["Verdict", detection.get('primary_result', {}).get('verdict', 'N/A')],
                ["Confidence", detection.get('primary_result', {}).get('confidence', 'N/A')],
                ["Ensemble Method", detection.get('ensemble_method', 'N/A')],
            ]
            
            table = Table(detection_data, colWidths=[2.5*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            content.append(table)
            content.append(Spacer(1, 20))
        
        # Metadata Results
        metadata = report.get("metadata")
        if metadata:
            content.append(Paragraph("Metadata Analysis", heading_style))
            
            file_info = metadata.get("file_info", {})
            content.append(Paragraph(f"<b>File:</b> {file_info.get('filename', 'N/A')}", styles['Normal']))
            content.append(Paragraph(f"<b>Dimensions:</b> {file_info.get('dimensions', 'N/A')}", styles['Normal']))
            content.append(Paragraph(f"<b>Format:</b> {file_info.get('format', 'N/A')}", styles['Normal']))
            content.append(Paragraph(f"<b>SHA-256:</b> {file_info.get('file_hash_sha256', 'N/A')[:32]}...", styles['Normal']))
            
            if metadata.get("anomalies"):
                content.append(Spacer(1, 10))
                content.append(Paragraph("<b>Anomalies Detected:</b>", styles['Normal']))
                for anomaly in metadata["anomalies"]:
                    content.append(Paragraph(f"• {anomaly}", styles['Normal']))
            
            content.append(Spacer(1, 20))
        
        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            content.append(Paragraph("Recommendations", heading_style))
            for rec in recommendations:
                content.append(Paragraph(f"• {rec}", styles['Normal']))
            content.append(Spacer(1, 20))
        
        # Methodology
        content.append(PageBreak())
        content.append(Paragraph("Methodology", heading_style))
        
        methodology = report.get("methodology", {})
        content.append(Paragraph(
            f"<b>Detection Approach:</b> {methodology.get('detection_approach', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 10))
        content.append(Paragraph(
            f"<b>Calibration:</b> {methodology.get('calibration', 'N/A')}",
            styles['Normal']
        ))
        content.append(Spacer(1, 10))
        
        if methodology.get("limitations"):
            content.append(Paragraph("<b>Limitations:</b>", styles['Normal']))
            for limitation in methodology["limitations"]:
                content.append(Paragraph(f"• {limitation}", styles['Normal']))
        
        # Footer
        content.append(Spacer(1, 40))
        content.append(Paragraph(
            "<i>This report was generated by ImageTrust - A Forensic Application "
            "for Identifying AI-Generated and Digitally Manipulated Images</i>",
            styles['Normal']
        ))
        
        # Build PDF
        doc.build(content)
        
        logger.info(f"PDF report exported to {output_path}")
        return output_path


class HTMLExporter:
    """Export reports to HTML format."""
    
    def export(
        self,
        report: dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        """
        Export report to HTML.
        
        Args:
            report: Report dictionary
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        html_content = self._generate_html(report)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to {output_path}")
        return output_path
    
    def _generate_html(self, report: dict[str, Any]) -> str:
        """Generate HTML content."""
        summary = report.get("summary", {})
        detection = report.get("detection", {})
        metadata = report.get("metadata", {})
        explainability = report.get("explainability", {})
        
        verdict = summary.get("verdict", "unknown")
        verdict_class = "danger" if verdict == "ai_generated" else "success" if verdict == "real" else "warning"
        
        # Build heatmap section
        heatmap_section = ""
        if explainability and explainability.get("heatmap_base64"):
            heatmap_section = f'''
            <div class="section">
                <h2>Explainability Analysis</h2>
                <div class="heatmap-container">
                    <img src="data:image/png;base64,{explainability['heatmap_base64']}" 
                         alt="Attention Heatmap" class="heatmap-image">
                    <p class="caption">Grad-CAM attention heatmap showing regions that influenced the detection.</p>
                </div>
            </div>
            '''
        
        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ImageTrust Forensic Report</title>
    <style>
        :root {{
            --primary: #1a1a2e;
            --secondary: #16213e;
            --accent: #0f3460;
            --success: #28a745;
            --danger: #dc3545;
            --warning: #ffc107;
            --light: #f8f9fa;
            --dark: #343a40;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: var(--light);
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .meta {{
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        
        .section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: var(--secondary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
        }}
        
        .verdict-box {{
            text-align: center;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        
        .verdict-box.success {{
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid var(--success);
        }}
        
        .verdict-box.danger {{
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            border: 2px solid var(--danger);
        }}
        
        .verdict-box.warning {{
            background: linear-gradient(135deg, #fff3cd, #ffeeba);
            border: 2px solid var(--warning);
        }}
        
        .verdict-text {{
            font-size: 2rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .probability {{
            font-size: 3rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .confidence {{
            font-size: 1.1rem;
            opacity: 0.8;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .metric-card {{
            background: var(--light);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: #666;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: var(--secondary);
            color: white;
        }}
        
        tr:hover {{
            background: var(--light);
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid var(--warning);
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
        }}
        
        .heatmap-container {{
            text-align: center;
        }}
        
        .heatmap-image {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        
        .caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .section {{
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 ImageTrust Forensic Report</h1>
            <div class="meta">
                Report ID: {report.get('report_id', 'N/A')} | 
                Generated: {report.get('generated_at', 'N/A')}
            </div>
        </header>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="verdict-box {verdict_class}">
                <div class="verdict-text">{verdict.replace('_', ' ')}</div>
                <div class="probability">{summary.get('ai_probability', 'N/A')}</div>
                <div class="confidence">Confidence: {summary.get('confidence_level', 'N/A')}</div>
            </div>
            <p>{summary.get('verdict_description', '')}</p>
        </div>
        
        <div class="section">
            <h2>Detection Results</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{detection.get('primary_result', {{}}).get('ai_probability', 0):.1%}</div>
                    <div class="metric-label">AI Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{detection.get('primary_result', {{}}).get('real_probability', 0):.1%}</div>
                    <div class="metric-label">Real Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(detection.get('models_used', []))}</div>
                    <div class="metric-label">Models Used</div>
                </div>
            </div>
        </div>
        
        {heatmap_section}
        
        <div class="section">
            <h2>Recommendations</h2>
            {''.join(f'<div class="alert alert-info">{rec}</div>' for rec in report.get('recommendations', []))}
        </div>
        
        <footer>
            <p>Generated by ImageTrust - A Forensic Application for Identifying AI-Generated and Digitally Manipulated Images</p>
        </footer>
    </div>
</body>
</html>
'''
