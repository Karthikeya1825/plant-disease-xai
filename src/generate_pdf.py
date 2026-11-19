"""
Publication-Grade PDF Report Generator:
Converts reports/COMPREHENSIVE_PROJECT_REPORT.md into a professionally styled,
typeset PDF document with embedded high-resolution figures, tables, and typography.
Uses Microsoft Edge / Chrome in headless print-to-pdf mode.
"""
import os
import re
import subprocess
from pathlib import Path
import markdown

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = PROJECT_DIR / "reports" / "COMPREHENSIVE_PROJECT_REPORT.md"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"
OUTPUT_HTML = PROJECT_DIR / "reports" / "report_styled.html"
OUTPUT_PDF = PROJECT_DIR / "reports" / "Plant_Disease_XAI_Comprehensive_Report.pdf"

def get_browser_path():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome was found for PDF rendering.")

def build_styled_html():
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "toc", "nl2br", "sane_lists"]
    )

    # Helper to convert image path to file URI
    def fig_uri(filename):
        fpath = (FIGURES_DIR / filename).resolve()
        return fpath.as_uri() if fpath.exists() else ""

    # Embed figures contextually into the HTML
    # 1. Dataset sample grid after Dataset Audit
    if (FIGURES_DIR / "dataset_sample_grid.png").exists():
        img_html = f'''
        <div class="figure-container">
            <img src="{fig_uri('dataset_sample_grid.png')}" alt="Dataset Sample Grid" style="max-width: 95%; border-radius: 6px;">
            <p class="figure-caption"><b>Figure 1:</b> Representative RGB sample grid from the PlantVillage color dataset across diverse crop species and pathology classes.</p>
        </div>
        '''
        html_body = re.sub(
            r'(<h3 id="22-class-taxonomy-and-distribution">.*?</h3>)',
            r'\1' + img_html,
            html_body,
            count=1,
            flags=re.DOTALL
        )

    # 2. Training curves in Step 3 / Step 4
    if (FIGURES_DIR / "mobilenet_v3_training_curves.png").exists():
        img_html = f'''
        <div class="figure-container">
            <img src="{fig_uri('mobilenet_v3_training_curves.png')}" alt="MobileNetV3 Training Curves" style="max-width: 85%; border-radius: 6px;">
            <p class="figure-caption"><b>Figure 2:</b> Training and validation loss/accuracy trajectory for MobileNetV3-Large under two-stage transfer learning with AdamW and Cosine Annealing.</p>
        </div>
        '''
        html_body = re.sub(
            r'(<h3 id="44-step-4-loss-functions-optimization-dynamics-scheduling">.*?</h3>)',
            img_html + r'\1',
            html_body,
            count=1,
            flags=re.DOTALL
        )

    # 3. Qualitative XAI Gallery in Step 5
    if (FIGURES_DIR / "xai_qualitative_gallery.png").exists():
        img_html = f'''
        <div class="figure-container">
            <img src="{fig_uri('xai_qualitative_gallery.png')}" alt="XAI Qualitative Gallery" style="max-width: 98%; border-radius: 6px;">
            <p class="figure-caption"><b>Figure 3:</b> Visual explainability panel comparing original inputs, Grad-CAM heatmaps, Grad-CAM overlays, and high-frequency Grad-CAM++ lesion localization overlays.</p>
        </div>
        '''
        html_body = re.sub(
            r'(<h3 id="46-step-6-quantitative-faithfulness-sanity-check-the-deletion-test">.*?</h3>)',
            img_html + r'\1',
            html_body,
            count=1,
            flags=re.DOTALL
        )

    # 4. Deletion test curve in Step 6
    if (FIGURES_DIR / "deletion_test_benchmark.png").exists():
        img_html = f'''
        <div class="figure-container">
            <img src="{fig_uri('deletion_test_benchmark.png')}" alt="Deletion Test Benchmark" style="max-width: 80%; border-radius: 6px;">
            <p class="figure-caption"><b>Figure 4:</b> Quantitative faithfulness validation (Deletion Test). Monotonic collapse in target prediction confidence under salient pixel masking vs. control random masking.</p>
        </div>
        '''
        html_body = re.sub(
            r'(<h3 id="47-step-7-isolated-test-set-evaluation-multi-metric-validation">.*?</h3>)',
            img_html + r'\1',
            html_body,
            count=1,
            flags=re.DOTALL
        )

    # 5. Confusion matrix & misclassified grid in Step 8
    if (FIGURES_DIR / "mobilenet_v3_confusion_matrix.png").exists():
        img_html = f'''
        <div class="figure-container">
            <div style="display: flex; gap: 10px; justify-content: center;">
                <img src="{fig_uri('mobilenet_v3_confusion_matrix.png')}" alt="Confusion Matrix" style="width: 48%; border-radius: 6px;">
                <img src="{fig_uri('mobilenet_v3_misclassified_samples.png')}" alt="Misclassified Grid" style="width: 48%; border-radius: 6px;">
            </div>
            <p class="figure-caption"><b>Figure 5:</b> (Left) Normalized Confusion Matrix heatmap for MobileNetV3 across all 38 classes. (Right) Error audit grid of misclassified samples with predicted vs. true ground-truth labels.</p>
        </div>
        '''
        html_body = re.sub(
            r'(<h2 id="5-system-prototyping-production-deployment">.*?</h2>)',
            img_html + r'\1',
            html_body,
            count=1,
            flags=re.DOTALL
        )

    # Publication CSS styling
    html_document = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Plant Disease Detection with Explainable AI - Comprehensive Technical Report</title>
    <!-- MathJax CDN for KaTeX/LaTeX math rendering -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
    </script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 8pt;
                color: #666;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.55;
            color: #212121;
            background-color: #FFFFFF;
            margin: 0;
            padding: 0;
        }}

        /* Header Cover Box */
        .header-cover {{
            background: linear-gradient(135deg, #1B5E20, #2E7D32);
            color: #FFFFFF;
            padding: 24px 28px;
            border-radius: 8px;
            margin-bottom: 24px;
            border-bottom: 4px solid #144A18;
        }}
        .header-cover h1 {{
            font-size: 19pt;
            margin: 0 0 8px 0;
            color: #FFFFFF;
            font-weight: 700;
            border: none;
            padding: 0;
        }}
        .header-cover .subtitle {{
            font-size: 11pt;
            color: #E8F5E9;
            margin: 0 0 14px 0;
            font-weight: 400;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px 16px;
            font-size: 8.5pt;
            border-top: 1px solid rgba(255,255,255,0.3);
            padding-top: 10px;
            color: #F1F8F4;
        }}

        h1, h2, h3, h4 {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1B5E20;
            font-weight: 600;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h1 {{
            font-size: 15pt;
            border-bottom: 2px solid #2E7D32;
            padding-bottom: 4px;
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        h2 {{
            font-size: 13pt;
            border-bottom: 1px solid #A5D6A7;
            padding-bottom: 3px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        h3 {{
            font-size: 11pt;
            color: #2E7D32;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        h4 {{
            font-size: 10pt;
            color: #388E3C;
            margin-top: 12px;
            margin-bottom: 6px;
        }}

        p {{
            margin-top: 0;
            margin-bottom: 9px;
            text-align: justify;
        }}

        /* Table of Contents */
        ul, ol {{
            margin-top: 0;
            margin-bottom: 10px;
            padding-left: 22px;
        }}
        li {{
            margin-bottom: 3px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 8.5pt;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        th, td {{
            padding: 6px 9px;
            border: 1px solid #CFD8DC;
            text-align: left;
        }}
        th {{
            background-color: #2E7D32;
            color: #FFFFFF;
            font-weight: 600;
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #F8FAF9;
        }}
        td:nth-child(n+2) {{
            text-align: center;
        }}

        /* Code Blocks */
        pre {{
            background-color: #F1F8F4;
            border: 1px solid #C8E6C9;
            border-radius: 5px;
            padding: 9px 12px;
            font-size: 8pt;
            font-family: Consolas, "Courier New", monospace;
            overflow-x: auto;
            margin: 10px 0;
            page-break-inside: avoid;
            break-inside: avoid;
            line-height: 1.4;
        }}
        code {{
            font-family: Consolas, "Courier New", monospace;
            font-size: 8.5pt;
            background-color: #F1F8F4;
            color: #1B5E20;
            padding: 1px 4px;
            border-radius: 3px;
            border: 1px solid #E0EAE2;
        }}
        pre code {{
            background-color: transparent;
            border: none;
            padding: 0;
            color: #212121;
        }}

        /* Blockquotes */
        blockquote {{
            background-color: #F1F8F4;
            border-left: 4px solid #2E7D32;
            margin: 10px 0;
            padding: 8px 14px;
            font-style: italic;
            color: #333333;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        /* Figures */
        .figure-container {{
            text-align: center;
            margin: 16px 0;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .figure-caption {{
            font-size: 8pt;
            color: #424242;
            margin-top: 6px;
            font-style: italic;
            text-align: center;
        }}

        hr {{
            border: none;
            border-top: 1px solid #E0E0E0;
            margin: 18px 0;
        }}

        /* Alert Callouts */
        .badge {{
            display: inline-block;
            padding: 2px 7px;
            border-radius: 10px;
            font-size: 7.5pt;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header-cover">
        <h1>🌿 Plant Disease Detection with Explainable AI (XAI)</h1>
        <div class="subtitle">A Comprehensive Empirical Study & Technical Reference Guide</div>
        <div class="meta-grid">
            <div><b>Academic Context:</b> Final-Year Deep Learning Research (Haridwar University #2)</div>
            <div><b>Dataset:</b> PlantVillage Color Subset (54,305 images / 38 classes)</div>
            <div><b>Architectures:</b> Baseline Scratch CNN, MobileNetV3-Large, EfficientNet-B0, ResNet-50</div>
            <div><b>Explainability & Verification:</b> Grad-CAM, Grad-CAM++, Deletion Ablation Testing</div>
        </div>
    </div>
    
    {html_body}
</body>
</html>'''

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_document)
    print(f"Styled HTML written to: {OUTPUT_HTML}")
    return OUTPUT_HTML

def render_pdf_with_browser():
    html_file = build_styled_html()
    browser_exe = get_browser_path()
    print(f"Using browser rendering engine: {browser_exe}")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=3000",
        f"--print-to-pdf={OUTPUT_PDF}",
        html_file.resolve().as_uri()
    ]

    print("Rendering publication-grade PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Browser stderr: {res.stderr}")
        raise RuntimeError(f"Browser failed to generate PDF. Return code: {res.returncode}")

    if OUTPUT_PDF.exists():
        size_kb = OUTPUT_PDF.stat().st_size / 1024
        print(f"SUCCESS: PDF successfully generated at:\n{OUTPUT_PDF} ({size_kb:.1f} KB)")
        return OUTPUT_PDF
    else:
        raise FileNotFoundError(f"PDF was not created at {OUTPUT_PDF}")

if __name__ == "__main__":
    render_pdf_with_browser()
