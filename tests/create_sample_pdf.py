"""
Script to create a sample PDF document for testing.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def create_sample_pdf(output_path):
    """Create a sample PDF document with text, tables, and structure."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    normal_style = styles["Normal"]
    
    # Create content elements
    elements = []
    
    # Title
    elements.append(Paragraph("Sample Document for Testing", title_style))
    elements.append(Spacer(1, 12))
    
    # Introduction
    elements.append(Paragraph("1. Introduction", heading_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "This is a sample PDF document created for testing the document processing "
        "pipeline in LightRAG. It contains text, tables, and structure that can be "
        "used to test various features of the system.", normal_style))
    elements.append(Spacer(1, 12))
    
    # Table of Contents (to test filtering)
    elements.append(Paragraph("Table of Contents", heading_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("1. Introduction........................... 1", normal_style))
    elements.append(Paragraph("2. Features.............................. 1", normal_style))
    elements.append(Paragraph("3. Sample Table.......................... 2", normal_style))
    elements.append(Paragraph("4. Conclusion............................ 2", normal_style))
    elements.append(Spacer(1, 12))
    
    # Features section
    elements.append(Paragraph("2. Features", heading_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "LightRAG provides several features for document processing and knowledge "
        "extraction:", normal_style))
    elements.append(Spacer(1, 6))
    
    features = [
        "Advanced PDF parsing with structure preservation",
        "Content filtering to remove headers, footers, and non-informative content",
        "Table extraction and conversion to Markdown",
        "Diagram and formula detection with placeholder replacement",
        "Schema-driven Knowledge Graph construction"
    ]
    
    for feature in features:
        elements.append(Paragraph(f"• {feature}", normal_style))
    
    elements.append(Spacer(1, 12))
    
    # Sample table
    elements.append(Paragraph("3. Sample Table", heading_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "The following table shows sample data that can be extracted by the table "
        "extraction component:", normal_style))
    elements.append(Spacer(1, 6))
    
    # Create a sample table
    data = [
        ["Model", "Type", "Parameters", "Performance"],
        ["GPT-4", "Transformer", "1.8T", "High"],
        ["Claude", "Transformer", "Unknown", "High"],
        ["Llama 2", "Transformer", "70B", "Medium"],
        ["Mistral", "Transformer", "7B", "Medium"]
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Conclusion
    elements.append(Paragraph("4. Conclusion", heading_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "This sample document demonstrates various features that can be processed "
        "by the LightRAG document processing pipeline. It includes structured text, "
        "a table of contents, a data table, and multiple sections with headings.", 
        normal_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "The document processing pipeline should be able to extract the main content, "
        "filter out the table of contents, and extract the data table for further "
        "processing.", normal_style))
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Sample PDF created at: {output_path}")

if __name__ == "__main__":
    # Create the fixtures directory if it doesn't exist
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Create the sample PDF
    output_path = os.path.join(fixtures_dir, "sample_doc.pdf")
    create_sample_pdf(output_path)
