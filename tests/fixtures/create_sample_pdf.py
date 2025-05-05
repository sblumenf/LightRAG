"""
Script to create a sample PDF file for testing.
"""
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_sample_pdf(output_path):
    """Create a sample PDF file with multiple pages and columns."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Page 1: Simple text
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, 10*inch, "Sample PDF Document")
    c.drawString(1*inch, 9.5*inch, "Page 1 - Simple Text")
    c.drawString(1*inch, 9*inch, "This is a sample PDF document created for testing.")
    c.drawString(1*inch, 8.5*inch, "It contains multiple pages with different layouts.")
    c.showPage()
    
    # Page 2: Two-column layout
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, 10*inch, "Page 2 - Two-Column Layout")
    
    # Left column
    c.drawString(1*inch, 9*inch, "Left Column - Line 1")
    c.drawString(1*inch, 8.5*inch, "Left Column - Line 2")
    c.drawString(1*inch, 8*inch, "Left Column - Line 3")
    
    # Right column
    c.drawString(4.5*inch, 9*inch, "Right Column - Line 1")
    c.drawString(4.5*inch, 8.5*inch, "Right Column - Line 2")
    c.drawString(4.5*inch, 8*inch, "Right Column - Line 3")
    c.showPage()
    
    # Page 3: Headers and footers
    c.setFont("Helvetica-Bold", 14)
    c.drawString(width/2 - 1.5*inch, 10.5*inch, "Page 3 - Headers and Footers")
    
    # Header
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, 10*inch, "HEADER: Sample Document")
    c.drawString(width - 2*inch, 10*inch, "Page 3")
    
    # Content
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, 9*inch, "This page has a header and footer.")
    c.drawString(1*inch, 8.5*inch, "The header contains the document title.")
    c.drawString(1*inch, 8*inch, "The footer contains the page number.")
    
    # Footer
    c.setFont("Helvetica", 10)
    c.drawString(width/2 - 0.5*inch, 0.5*inch, "Page 3 of 3")
    c.showPage()
    
    # Save the PDF
    c.save()
    
    return output_path

if __name__ == "__main__":
    # Create the fixtures directory if it doesn't exist
    fixtures_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the sample PDF
    output_path = os.path.join(fixtures_dir, "sample.pdf")
    create_sample_pdf(output_path)
    print(f"Sample PDF created at: {output_path}")
