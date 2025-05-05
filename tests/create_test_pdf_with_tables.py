"""
Script to create a test PDF with tables for testing the table extractor.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def create_test_pdf_with_tables(output_path):
    """Create a test PDF with tables for testing the table extractor."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create content elements
    elements = []
    
    # Title
    elements.append(Paragraph("Test PDF with Tables", styles["Title"]))
    elements.append(Spacer(1, 12))
    
    # Introduction
    elements.append(Paragraph("This is a test PDF document with tables for testing the table extractor.", styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Create a simple table
    data = [
        ["Header 1", "Header 2", "Header 3"],
        ["Row 1, Col 1", "Row 1, Col 2", "Row 1, Col 3"],
        ["Row 2, Col 1", "Row 2, Col 2", "Row 2, Col 3"]
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
    elements.append(Spacer(1, 24))
    
    # Create a more complex table
    data2 = [
        ["Model", "Type", "Parameters", "Performance"],
        ["GPT-4", "Transformer", "1.8T", "High"],
        ["Claude", "Transformer", "Unknown", "High"],
        ["Llama 2", "Transformer", "70B", "Medium"],
        ["Mistral", "Transformer", "7B", "Medium"]
    ]
    
    table2 = Table(data2)
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table2)
    
    # Build the PDF
    doc.build(elements)
    
    print(f"Test PDF with tables created at: {output_path}")

if __name__ == "__main__":
    # Create the fixtures directory if it doesn't exist
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Create the test PDF
    output_path = os.path.join(fixtures_dir, "test_tables.pdf")
    create_test_pdf_with_tables(output_path)
