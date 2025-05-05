"""
Create a test PDF file with diagrams and formulas for testing.
"""
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import blue, red, green, black

def create_test_pdf(output_path):
    """Create a test PDF with diagrams and formulas."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Add a title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, "Test PDF with Diagrams and Formulas")
    
    # Add some text
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, height - 1.5*inch, "This is a test document for extracting diagrams and formulas.")
    
    # Add a simple diagram (flowchart)
    c.setStrokeColor(blue)
    c.setFillColor(blue)
    # Box 1
    c.rect(2*inch, height - 3*inch, 2*inch, 0.75*inch, stroke=1, fill=0)
    c.setFont("Helvetica", 10)
    c.setFillColor(black)
    c.drawString(2.5*inch, height - 2.7*inch, "Start")
    
    # Arrow 1
    c.setStrokeColor(blue)
    c.line(3*inch, height - 3*inch, 3*inch, height - 3.25*inch)
    c.line(3*inch, height - 3.25*inch, 3.1*inch, height - 3.15*inch)
    c.line(3*inch, height - 3.25*inch, 2.9*inch, height - 3.15*inch)
    
    # Box 2
    c.rect(2*inch, height - 4*inch, 2*inch, 0.75*inch, stroke=1, fill=0)
    c.setFillColor(black)
    c.drawString(2.3*inch, height - 3.7*inch, "Process Data")
    
    # Arrow 2
    c.setStrokeColor(blue)
    c.line(3*inch, height - 4*inch, 3*inch, height - 4.25*inch)
    c.line(3*inch, height - 4.25*inch, 3.1*inch, height - 4.15*inch)
    c.line(3*inch, height - 4.25*inch, 2.9*inch, height - 4.15*inch)
    
    # Box 3
    c.rect(2*inch, height - 5*inch, 2*inch, 0.75*inch, stroke=1, fill=0)
    c.setFillColor(black)
    c.drawString(2.5*inch, height - 4.7*inch, "End")
    
    # Add a caption for the diagram
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(2*inch, height - 5.25*inch, "Figure 1: Simple Flowchart Diagram")
    
    # Add some more text
    c.setFont("Helvetica", 12)
    c.drawString(1*inch, height - 6*inch, "Below are some mathematical formulas:")
    
    # Add formulas
    c.setFont("Helvetica", 12)
    c.drawString(1.5*inch, height - 6.5*inch, "Einstein's famous equation: E = mc²")
    c.drawString(1.5*inch, height - 7*inch, "Pythagorean theorem: a² + b² = c²")
    c.drawString(1.5*inch, height - 7.5*inch, "Quadratic formula: x = (-b ± √(b² - 4ac)) / 2a")
    
    # Add a simple bar chart
    c.setStrokeColor(black)
    c.setFillColor(black)
    c.drawString(5*inch, height - 3*inch, "Simple Bar Chart")
    
    # Draw axes
    c.line(5*inch, height - 5*inch, 5*inch, height - 3.5*inch)  # Y-axis
    c.line(5*inch, height - 5*inch, 7*inch, height - 5*inch)    # X-axis
    
    # Draw bars
    bar_width = 0.3*inch
    c.setFillColor(red)
    c.rect(5.2*inch, height - 5*inch, bar_width, 0.5*inch, stroke=1, fill=1)
    c.setFillColor(green)
    c.rect(5.6*inch, height - 5*inch, bar_width, 0.8*inch, stroke=1, fill=1)
    c.setFillColor(blue)
    c.rect(6*inch, height - 5*inch, bar_width, 0.3*inch, stroke=1, fill=1)
    c.setFillColor(red)
    c.rect(6.4*inch, height - 5*inch, bar_width, 0.7*inch, stroke=1, fill=1)
    
    # Add labels
    c.setFillColor(black)
    c.setFont("Helvetica", 8)
    c.drawString(5.2*inch, height - 5.2*inch, "A")
    c.drawString(5.6*inch, height - 5.2*inch, "B")
    c.drawString(6*inch, height - 5.2*inch, "C")
    c.drawString(6.4*inch, height - 5.2*inch, "D")
    
    # Add a caption for the chart
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(5*inch, height - 5.4*inch, "Figure 2: Simple Bar Chart")
    
    # Save the PDF
    c.save()
    print(f"Created test PDF at {output_path}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "sample_doc.pdf")
    create_test_pdf(output_path)
