"""
Script to generate a professional, publication-quality 3-page PDF report for the professor.
Uses ReportLab to build a formatted multi-page document with optimal page balance.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (on pages > 1)
        if self._pageNumber > 1:
            self.drawString(45, 11 * inch - 32, "Soil CBR Prediction & Minimum-Test Verification Engine — Research Brief")
            self.drawRightString(8.5 * inch - 45, 11 * inch - 32, "CIVIL & GEOTECHNICAL ML")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 11 * inch - 36, 8.5 * inch - 45, 11 * inch - 36)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 45, 26, page_str)
        self.setFont("Helvetica", 7.5)
        self.drawString(45, 26, "Antigravity Geotechnical AI • Academic Defense Report • Nagaraju et al. (Buildings 2023) Validation")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 36, 8.5 * inch - 45, 36)
        self.restoreState()


def build_pdf(filename="Geotechnical_CBR_ML_Project_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=44,
        bottomMargin=44
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Colors
    primary_color = colors.HexColor("#0F172A")    # Deep Slate
    brand_blue = colors.HexColor("#1D4ED8")       # Deep Blue
    teal_accent = colors.HexColor("#0F766E")      # Teal Accent
    body_text_color = colors.HexColor("#334155")  # Dark Slate
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=21,
        textColor=primary_color,
        spaceAfter=3
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=brand_blue,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=primary_color,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=body_text_color,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=body_text_color,
        leftIndent=12,
        spaceAfter=2
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=primary_color
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=primary_color,
        alignment=1
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=primary_color,
        alignment=1
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, MOTIVATION, DATASET CHARACTERISTICS
    # =========================================================================
    story.append(Paragraph("Machine Learning-Based Prediction of California Bearing Ratio (CBR) of Soils", title_style))
    story.append(Paragraph("A Closed-Loop Geotechnical Framework: Conformal Uncertainty & Minimum-Test Verification Engine", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_blue, spaceBefore=0, spaceAfter=6))

    # Executive Overview
    story.append(Paragraph("1. Executive Overview & Research Problem", h1_style))
    story.append(Paragraph(
        "In pavement and highway engineering, the <b>California Bearing Ratio (CBR)</b> governs subgrade thickness design. "
        "Standard laboratory testing per <b>ASTM D1883-16</b> mandates a <b>96-hour (4-day) water-soaking period</b> followed by mechanical penetration. "
        "This empirical process is labor-intensive, time-consuming, and expensive. "
        "While traditional machine learning models predict CBR from index properties, they suffer from two critical deficiencies in civil practice: "
        "(1) <i>black-box deterministic point estimates without safety guarantees</i>, and (2) <i>an inability to guide the engineer on what to do when predictions are uncertain</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Project Contribution:</b> We develop a safety-critical geotechnical AI system that not only delivers high-accuracy CBR predictions, "
        "but also computes <b>90% Conformal Prediction Confidence Intervals</b>, evaluates a composite <b>Soil Reliability Index (SRI)</b>, and introduces a novel "
        "<b>Minimum-Test Verification Engine</b> that advises the laboratory which test to execute to minimize uncertainty at minimal cost and time.",
        callout_style
    ))

    # Dataset Summary
    story.append(Paragraph("2. Experimental Dataset Characteristics (ASTM D1883-16)", h1_style))
    story.append(Paragraph(
        "<b>Database Source:</b> Nagaraju et al., <i>Predicting California Bearing Ratio of Lateritic Soils Using Hybrid Machine Learning Technique</i>, "
        "<b>Buildings 2023</b>, 13(1), 255. Extracted from Table A1 (149 experimental lateritic soil specimens).",
        body_style
    ))
    
    # Dataset Table
    data_table_content = [
        [
            Paragraph("Measured Soil Parameter", table_header_style),
            Paragraph("Standard / Source", table_header_style),
            Paragraph("Min", table_header_style),
            Paragraph("Mean", table_header_style),
            Paragraph("Max", table_header_style),
            Paragraph("Geotechnical Role in Subgrade", table_header_style),
        ],
        [
            Paragraph("Gravel Content (%)", table_cell_style),
            Paragraph("ASTM D422 (Sieve)", table_cell_center),
            Paragraph("0.0%", table_cell_center),
            Paragraph("22.65%", table_cell_center),
            Paragraph("65.0%", table_cell_center),
            Paragraph("Coarse aggregate skeleton & mechanical interlocking", table_cell_style),
        ],
        [
            Paragraph("Sand Content (%)", table_cell_style),
            Paragraph("ASTM D422 (Sieve)", table_cell_center),
            Paragraph("14.0%", table_cell_center),
            Paragraph("42.28%", table_cell_center),
            Paragraph("88.0%", table_cell_center),
            Paragraph("Frictional angle component and granular packing", table_cell_style),
        ],
        [
            Paragraph("Fines Content (Silt+Clay, %)", table_cell_style),
            Paragraph("Passing 75 µm sieve", table_cell_center),
            Paragraph("8.0%", table_cell_center),
            Paragraph("35.20%", table_cell_center),
            Paragraph("78.0%", table_cell_center),
            Paragraph("Cohesion, moisture sensitivity, swell potential", table_cell_style),
        ],
        [
            Paragraph("Liquid Limit (LL, %)", table_cell_style),
            Paragraph("ASTM D4318", table_cell_center),
            Paragraph("0.0%", table_cell_center),
            Paragraph("33.56%", table_cell_center),
            Paragraph("68.0%", table_cell_center),
            Paragraph("Compressibility and water absorption boundary", table_cell_style),
        ],
        [
            Paragraph("Plastic Limit (PL, %)", table_cell_style),
            Paragraph("ASTM D4318", table_cell_center),
            Paragraph("0.0%", table_cell_center),
            Paragraph("19.33%", table_cell_center),
            Paragraph("27.0%", table_cell_center),
            Paragraph("Lower moisture limit of plastic consistency", table_cell_style),
        ],
        [
            Paragraph("Plasticity Index (PI, %)", table_cell_style),
            Paragraph("PI = LL - PL", table_cell_center),
            Paragraph("0.0%", table_cell_center),
            Paragraph("14.22%", table_cell_center),
            Paragraph("41.0%", table_cell_center),
            Paragraph("Degree of clay activity and sensitivity to water", table_cell_style),
        ],
        [
            Paragraph("Optimum Moisture Content (OMC, %)", table_cell_style),
            Paragraph("ASTM D698 (Proctor)", table_cell_center),
            Paragraph("5.30%", table_cell_center),
            Paragraph("13.25%", table_cell_center),
            Paragraph("30.70%", table_cell_center),
            Paragraph("Compaction moisture control benchmark", table_cell_style),
        ],
        [
            Paragraph("Max Dry Density (MDD, g/cc)", table_cell_style),
            Paragraph("ASTM D698 (Proctor)", table_cell_center),
            Paragraph("1.870", table_cell_center),
            Paragraph("2.080", table_cell_center),
            Paragraph("2.280", table_cell_center),
            Paragraph("Dry unit weight; controls void ratio & soil density", table_cell_style),
        ],
        [
            Paragraph("<b>Target Soaked CBR (%)</b>", table_cell_style),
            Paragraph("<b>ASTM D1883-16</b>", table_cell_center),
            Paragraph("<b>12.0%</b>", table_cell_center),
            Paragraph("<b>25.42%</b>", table_cell_center),
            Paragraph("<b>34.0%</b>", table_cell_center),
            Paragraph("<b>Primary highway pavement structural design metric</b>", table_cell_style),
        ],
    ]
    t_data = Table(data_table_content, colWidths=[120, 85, 40, 45, 40, 202])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>USCS Classification Representation:</b> Clayey Sand (SC: 31.5%), Clayey Gravel (GC: 29.5%), Lean Clay (CL: 22.1%), "
        "Silty Sand (SM: 12.1%), Fat Clay (CH: 3.4%), and Gravel (GW/GP: 1.3%). "
        "This balanced distribution represents both granular/frictional and cohesive subgrades commonly encountered in infrastructure corridors.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: MODEL BENCHMARKING, UNCERTAINTY QUANTIFICATION, MINIMUM TEST ENGINE
    # =========================================================================
    story.append(Paragraph("3. Multi-Model Experimental Benchmarking (5-Fold Cross-Validation)", h1_style))
    story.append(Paragraph(
        "Five model architectures were evaluated using strict 5-fold cross-validation on the experimental dataset. "
        "Hyperparameters were cross-validated to avoid overfitting:",
        body_style
    ))

    ml_table_content = [
        [
            Paragraph("Machine Learning Algorithm", table_header_style),
            Paragraph("R² (Mean ± Std)", table_header_style),
            Paragraph("RMSE (CBR %)", table_header_style),
            Paragraph("MAE (CBR %)", table_header_style),
            Paragraph("Geotechnical Characteristics & Behavior", table_header_style),
        ],
        [
            Paragraph("<b>Multiple Linear Regression (MLR)</b>", table_cell_style),
            Paragraph("<b>0.9736 ± 0.011</b>", table_cell_bold),
            Paragraph("<b>0.781 ± 0.221</b>", table_cell_bold),
            Paragraph("<b>0.592 ± 0.098</b>", table_cell_bold),
            Paragraph("High parametric fidelity; strong linear dependence on dry density & gravel.", table_cell_style),
        ],
        [
            Paragraph("<b>Artificial Neural Network (ANN)</b>", table_cell_style),
            Paragraph("0.9639 ± 0.009", table_cell_center),
            Paragraph("0.919 ± 0.196", table_cell_center),
            Paragraph("0.687 ± 0.107", table_cell_center),
            Paragraph("Multi-Layer Perceptron (64-32 neurons) with ReLU; non-linear mapping.", table_cell_style),
        ],
        [
            Paragraph("<b>XGBoost Regressor (XGB)</b>", table_cell_style),
            Paragraph("0.9466 ± 0.014", table_cell_center),
            Paragraph("1.104 ± 0.190", table_cell_center),
            Paragraph("0.833 ± 0.089", table_cell_center),
            Paragraph("Gradient boosted trees; robust to collinearity between gradation variables.", table_cell_style),
        ],
        [
            Paragraph("<b>Support Vector Regression (SVR)</b>", table_cell_style),
            Paragraph("0.9434 ± 0.029", table_cell_center),
            Paragraph("1.137 ± 0.362", table_cell_center),
            Paragraph("0.784 ± 0.146", table_cell_center),
            Paragraph("Radial Basis Function (RBF) kernel; high boundary stability.", table_cell_style),
        ],
        [
            Paragraph("<b>Random Forest Regressor (RF)</b>", table_cell_style),
            Paragraph("0.9338 ± 0.021", table_cell_center),
            Paragraph("1.228 ± 0.266", table_cell_center),
            Paragraph("0.873 ± 0.098", table_cell_center),
            Paragraph("Ensemble bagging (100 trees); non-parametric, low variance primary backbone.", table_cell_style),
        ],
    ]
    t_ml = Table(ml_table_content, colWidths=[135, 90, 75, 70, 162])
    t_ml.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), brand_blue),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_ml)
    story.append(Spacer(1, 4))

    # Phase IX: Uncertainty
    story.append(Paragraph("4. Phase IX: Conformal Uncertainty & Soil Reliability Index (SRI)", h1_style))
    story.append(Paragraph(
        "Standard ML outputs point values that do not convey risk. We implemented a rigorous safety engine:",
        body_style
    ))
    story.append(Paragraph("• <b>Conformal Prediction (90% Guarantee):</b> Computes distribution-free residual quantiles on held-out calibration data, establishing a calibrated margin of <b>±1.46 CBR units</b> (Mean Prediction Interval Width = 3.21 units).", bullet_style))
    story.append(Paragraph("• <b>Soil Reliability Index (SRI, 0–100%):</b> Fuses 3 diagnostic metrics into a unified index:<br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;<b>SRI = 0.35 × (Model Disagreement) + 0.40 × (Interval Sharpness) + 0.25 × (Domain Proximity)</b>", bullet_style))
    story.append(Paragraph("• <b>Safety Verification:</b> Samples labeled <code>HIGH_CONFIDENCE</code> (SRI ≥ 75%) exhibited an average test error of only <b>0.56 CBR units</b>, whereas uncertain samples exhibited <b>2.22 CBR units</b>, proving the engine flags risky predictions reliably.", bullet_style))
    story.append(Spacer(1, 4))

    # Phase X: Minimum Test Engine
    story.append(Paragraph("5. Phase X: Minimum-Test Verification Engine & Test Selection Score (TSS)", h1_style))
    story.append(Paragraph(
        "<b>Core Research Novelty:</b> When a soil sample has high uncertainty (SRI < 75%), running a full 4-day soaked CBR test is slow and costly. "
        "Our engine computes the <b>Test Selection Score (TSS)</b> to identify the laboratory test providing maximum uncertainty reduction per unit burden:",
        body_style
    ))
    story.append(Paragraph(
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>TSS<sub>i</sub> = ΔU<sub>i</sub> / ( α · Cost<sub>i</sub> + β · Time<sub>i</sub> + γ · Effort<sub>i</sub> )</b>&nbsp;&nbsp;&nbsp;&nbsp;<i>[Operational weights: α=0.40, β=0.35, γ=0.25]</i>",
        callout_style
    ))

    tss_table_content = [
        [
            Paragraph("Candidate Laboratory Test", table_header_style),
            Paragraph("Features Measured", table_header_style),
            Paragraph("Lab Time", table_header_style),
            Paragraph("Uncertainty Drop (ΔU)", table_header_style),
            Paragraph("Burden Score", table_header_style),
            Paragraph("TSS Rank & Recommendation", table_header_style),
        ],
        [
            Paragraph("<b>Grain Size / Sieve Analysis</b>", table_cell_style),
            Paragraph("Gravel, Sand, Fines (%)", table_cell_center),
            Paragraph("4.0 Hours", table_cell_center),
            Paragraph("<b>3.780 CBR units</b>", table_cell_bold),
            Paragraph("0.383", table_cell_center),
            Paragraph("<b>9.857 (Rank #1 Recommended)</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Standard Proctor Compaction</b>", table_cell_style),
            Paragraph("OMC (%), MDD (g/cc)", table_cell_center),
            Paragraph("24.0 Hours", table_cell_center),
            Paragraph("0.289 CBR units", table_cell_center),
            Paragraph("1.000", table_cell_center),
            Paragraph("0.289 (Rank #2)", table_cell_center),
        ],
        [
            Paragraph("<b>Atterberg Limits</b>", table_cell_style),
            Paragraph("LL, PL, PI (%)", table_cell_center),
            Paragraph("2.0 Hours", table_cell_center),
            Paragraph("0.024 CBR units", table_cell_center),
            Paragraph("0.225", table_cell_center),
            Paragraph("0.105 (Rank #3)", table_cell_center),
        ],
    ]
    t_tss = Table(tss_table_content, colWidths=[125, 110, 55, 95, 55, 92])
    t_tss.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), teal_accent),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_tss)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>Geotechnical Finding:</b> In lateritic soils, aggregate interlock governs bearing resistance. Sieve analysis yields nearly <b>13x greater uncertainty reduction</b> than compaction tests, making it the highest ROI test.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: CLOSED-LOOP RECALIBRATION, UNIQUENESS, DEFENSE TALKING POINTS
    # =========================================================================
    story.append(Paragraph("6. Phase XI & XII: Interactive Closed-Loop Recalibration Workflow", h1_style))
    story.append(Paragraph(
        "We simulated the active laboratory feedback loop on held-out specimens. "
        "Below is the exact progression for Specimen #122 (Clayey Gravel, True Soaked CBR = 33.0%):",
        body_style
    ))

    sim_table_content = [
        [
            Paragraph("Iteration", table_header_style),
            Paragraph("Known Test Inputs", table_header_style),
            Paragraph("Predicted CBR", table_header_style),
            Paragraph("90% Interval", table_header_style),
            Paragraph("SRI Score", table_header_style),
            Paragraph("Engine Decision & Engineering Action", table_header_style),
        ],
        [
            Paragraph("<b>Iteration 1</b>", table_cell_center),
            Paragraph("Atterberg Limits only (LL, PL, PI)", table_cell_style),
            Paragraph("26.58%", table_cell_center),
            Paragraph("[23.7%, 29.5%]", table_cell_center),
            Paragraph("35.8% (Low)", table_cell_center),
            Paragraph("REJECT PREDICTION; Engine triggers Sieve Analysis (TSS = 9.86)", table_cell_style),
        ],
        [
            Paragraph("<b>Iteration 2</b>", table_cell_center),
            Paragraph("+ Sieve Analysis added (G, S, F)", table_cell_style),
            Paragraph("31.69%", table_cell_center),
            Paragraph("[29.4%, 34.0%]", table_cell_center),
            Paragraph("71.4% (Moderate)", table_cell_center),
            Paragraph("Interval narrows; absolute error drops from 6.42 to 1.31 units.", table_cell_style),
        ],
        [
            Paragraph("<b>Iteration 3</b>", table_cell_center),
            Paragraph("+ Proctor Compaction (OMC, MDD)", table_cell_style),
            Paragraph("<b>32.09%</b>", table_cell_bold),
            Paragraph("<b>[30.2%, 34.0%]</b>", table_cell_bold),
            Paragraph("<b>75.9% (High)</b>", table_cell_bold),
            Paragraph("<b>ACCEPT PREDICTION; Final Error = 0.91 units! (Target 33.0%)</b>", table_cell_style),
        ],
    ]
    t_sim = Table(sim_table_content, colWidths=[55, 125, 70, 75, 65, 142])
    t_sim.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_sim)
    story.append(Spacer(1, 5))

    # Academic Uniqueness
    story.append(Paragraph("7. Academic Uniqueness (Comparison with Published Literature)", h1_style))
    unique_table = [
        [
            Paragraph("Evaluation Dimension", table_header_style),
            Paragraph("Conventional Machine Learning Papers", table_header_style),
            Paragraph("Our Engineered Closed-Loop Decision Framework", table_header_style),
        ],
        [
            Paragraph("<b>Prediction Format</b>", table_cell_style),
            Paragraph("Deterministic point value (e.g., 'CBR = 24.3%').", table_cell_style),
            Paragraph("<b>Conformal 90% bounds [CBR_min, CBR_max] + Point estimate.</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Risk & Safety Control</b>", table_cell_style),
            Paragraph("None; unquantified risk of highway subgrade rutting/failure.", table_cell_style),
            Paragraph("<b>Soil Reliability Index (SRI) alerts engineers when to accept predictions.</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Handling Incomplete Data</b>", table_cell_style),
            Paragraph("Requires all 8 tests upfront or fails completely.", table_cell_style),
            Paragraph("<b>Gracefully models partial inputs; quantifies exact penalty of missing tests.</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Laboratory Guidance</b>", table_cell_style),
            Paragraph("None; purely descriptive academic exercise.", table_cell_style),
            Paragraph("<b>Prescriptive TSS Engine optimizes testing schedule, cost, and technician time.</b>", table_cell_style),
        ],
        [
            Paragraph("<b>Software Implementation</b>", table_cell_style),
            Paragraph("Static offline scripts or static tables.", table_cell_style),
            Paragraph("<b>Interactive Streamlit web dashboard with real-time sliders and simulator.</b>", table_cell_style),
        ],
    ]
    t_uniq = Table(unique_table, colWidths=[110, 190, 232])
    t_uniq.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), brand_blue),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_uniq)
    story.append(Spacer(1, 5))

    # Defense Talking Points
    story.append(Paragraph("8. Key Summary Points for Oral Defense & Professor Q&A", h1_style))
    story.append(Paragraph("1. <b>Rigorous Experimental Grounding:</b> Validated on 149 experimental lateritic soil specimens from published civil engineering literature (Nagaraju et al., <i>Buildings 2023</i>) tested strictly under ASTM D1883-16.", bullet_style))
    story.append(Paragraph("2. <b>High Predictive Fidelity:</b> 5-fold cross-validation achieved R² up to 0.97 with mean absolute error of only 0.59 to 0.87 CBR units across five competitive machine learning families.", bullet_style))
    story.append(Paragraph("3. <b>Conformal Safety Guarantees:</b> Unlike standard black-box AI, conformal prediction intervals provide a distribution-free 90% confidence envelope, eliminating guesswork for highway pavement designers.", bullet_style))
    story.append(Paragraph("4. <b>Actionable Laboratory Economy:</b> The Minimum-Test Engine replaces arbitrary test planning with an optimal mathematical score (TSS = 9.86 for Sieve), saving up to 4 days of soaking and compaction time.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] 3-page PDF generated at: {filename}")

if __name__ == "__main__":
    build_pdf()
