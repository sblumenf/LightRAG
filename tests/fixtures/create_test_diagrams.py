"""
Create test diagrams for diagram entity extraction testing.

This script generates synthetic diagrams that can be used for testing the 
diagram entity extraction feature.
"""
import os
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

# Paths
BASE_DIR = Path(__file__).parent
DIAGRAMS_DIR = BASE_DIR / "diagrams"

# Ensure diagrams directory exists
DIAGRAMS_DIR.mkdir(exist_ok=True)

def create_architecture_diagram():
    """Create a simple architecture diagram with components and connections."""
    # Create new white image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font_path = BASE_DIR / "fonts" / "DejaVuSans.ttf"
        title_font = ImageFont.truetype(str(font_path), 20)
        label_font = ImageFont.truetype(str(font_path), 16)
    except IOError:
        print("Font file not found, using default font")
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2 - 150, 20), "System Architecture Diagram", fill='black', font=title_font)
    
    # Draw components
    components = [
        {"name": "Frontend", "x": 200, "y": 150, "width": 150, "height": 80, "color": "lightblue"},
        {"name": "API Gateway", "x": 400, "y": 150, "width": 150, "height": 80, "color": "lightgreen"},
        {"name": "User Service", "x": 200, "y": 300, "width": 150, "height": 80, "color": "lightyellow"},
        {"name": "Content Service", "x": 400, "y": 300, "width": 150, "height": 80, "color": "lightyellow"},
        {"name": "Database", "x": 600, "y": 300, "width": 150, "height": 80, "color": "lightcoral"}
    ]
    
    # Draw each component
    for component in components:
        # Draw rectangle
        draw.rectangle(
            [(component["x"], component["y"]), 
             (component["x"] + component["width"], component["y"] + component["height"])],
            fill=component["color"],
            outline="black",
            width=2
        )
        
        # Draw text
        text_width = draw.textlength(component["name"], font=label_font)
        text_x = component["x"] + (component["width"] - text_width) // 2
        text_y = component["y"] + (component["height"] - 16) // 2
        draw.text((text_x, text_y), component["name"], fill='black', font=label_font)
    
    # Draw connections
    connections = [
        {"from": 0, "to": 1, "label": "HTTP"},  # Frontend to API Gateway
        {"from": 1, "to": 2, "label": "REST"},  # API Gateway to User Service
        {"from": 1, "to": 3, "label": "REST"},  # API Gateway to Content Service
        {"from": 2, "to": 4, "label": "SQL"},   # User Service to Database
        {"from": 3, "to": 4, "label": "SQL"}    # Content Service to Database
    ]
    
    # Draw each connection
    for conn in connections:
        from_comp = components[conn["from"]]
        to_comp = components[conn["to"]]
        
        # Calculate connection points
        from_x = from_comp["x"] + from_comp["width"]
        from_y = from_comp["y"] + from_comp["height"] // 2
        to_x = to_comp["x"]
        to_y = to_comp["y"] + to_comp["height"] // 2
        
        # Handle horizontal connections
        if abs(from_y - to_y) < 5:
            # Draw horizontal line
            draw.line([(from_x, from_y), (to_x, to_y)], fill='black', width=2)
            # Draw label in the middle
            mid_x = (from_x + to_x) // 2
            mid_y = from_y - 20
            draw.text((mid_x, mid_y), conn["label"], fill='black', font=label_font)
        else:
            # Calculate midpoints for bent line
            mid_x1 = from_x + 20
            mid_x2 = to_x - 20
            
            # Draw bent line
            draw.line([(from_x, from_y), (mid_x1, from_y), (mid_x1, to_y), (to_x, to_y)], fill='black', width=2)
            
            # Draw label at the midpoint
            label_x = mid_x1 + 5
            label_y = (from_y + to_y) // 2
            draw.text((label_x, label_y), conn["label"], fill='black', font=label_font)
    
    # Save the image
    output_path = DIAGRAMS_DIR / "system_architecture.png"
    img.save(output_path)
    print(f"Architecture diagram saved to {output_path}")
    return output_path

def create_flowchart_diagram():
    """Create a simple flowchart diagram."""
    # Create new white image
    width, height = 600, 800
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font_path = BASE_DIR / "fonts" / "DejaVuSans.ttf"
        title_font = ImageFont.truetype(str(font_path), 20)
        label_font = ImageFont.truetype(str(font_path), 16)
    except IOError:
        print("Font file not found, using default font")
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2 - 100, 20), "User Login Process", fill='black', font=title_font)
    
    # Draw flowchart elements
    elements = [
        {"type": "start", "name": "Start", "x": 300, "y": 100, "width": 120, "height": 50, "color": "lightgreen"},
        {"type": "process", "name": "User Login Request", "x": 300, "y": 180, "width": 200, "height": 60, "color": "lightblue"},
        {"type": "decision", "name": "Valid Credentials?", "x": 300, "y": 280, "width": 160, "height": 80, "color": "lightyellow"},
        {"type": "process", "name": "Generate Token", "x": 150, "y": 400, "width": 180, "height": 60, "color": "lightblue"},
        {"type": "process", "name": "Return Error", "x": 450, "y": 400, "width": 180, "height": 60, "color": "lightcoral"},
        {"type": "process", "name": "Return Success", "x": 150, "y": 500, "width": 180, "height": 60, "color": "lightblue"},
        {"type": "end", "name": "End", "x": 300, "y": 600, "width": 120, "height": 50, "color": "lightgreen"}
    ]
    
    # Draw each element
    for element in elements:
        # Handle different shapes
        if element["type"] == "decision":
            # Draw diamond
            diamond_points = [
                (element["x"] + element["width"]//2, element["y"]),  # top
                (element["x"] + element["width"], element["y"] + element["height"]//2),  # right
                (element["x"] + element["width"]//2, element["y"] + element["height"]),  # bottom
                (element["x"], element["y"] + element["height"]//2)  # left
            ]
            draw.polygon(diamond_points, fill=element["color"], outline="black")
            
            # Draw text (centered)
            lines = element["name"].split()
            line_height = 20
            start_y = element["y"] + (element["height"] - len(lines) * line_height) // 2
            
            for i, line in enumerate(lines):
                text_width = draw.textlength(line, font=label_font)
                text_x = element["x"] + (element["width"] - text_width) // 2
                draw.text((text_x, start_y + i * line_height), line, fill='black', font=label_font)
                
        elif element["type"] in ["start", "end"]:
            # Draw rounded rectangle (approximated with an ellipse)
            draw.ellipse(
                [(element["x"], element["y"]), 
                 (element["x"] + element["width"], element["y"] + element["height"])],
                fill=element["color"],
                outline="black",
                width=2
            )
            
            # Draw text
            text_width = draw.textlength(element["name"], font=label_font)
            text_x = element["x"] + (element["width"] - text_width) // 2
            text_y = element["y"] + (element["height"] - 16) // 2
            draw.text((text_x, text_y), element["name"], fill='black', font=label_font)
            
        else:  # process
            # Draw rectangle
            draw.rectangle(
                [(element["x"], element["y"]), 
                 (element["x"] + element["width"], element["y"] + element["height"])],
                fill=element["color"],
                outline="black",
                width=2
            )
            
            # Draw text
            text_width = draw.textlength(element["name"], font=label_font)
            text_x = element["x"] + (element["width"] - text_width) // 2
            text_y = element["y"] + (element["height"] - 16) // 2
            draw.text((text_x, text_y), element["name"], fill='black', font=label_font)
    
    # Draw connections
    connections = [
        {"from": 0, "to": 1, "label": ""},  # Start to Login Request
        {"from": 1, "to": 2, "label": ""},  # Login Request to Valid Credentials
        {"from": 2, "to": 3, "label": "Yes"},  # Valid Credentials to Generate Token
        {"from": 2, "to": 4, "label": "No"},  # Valid Credentials to Return Error
        {"from": 3, "to": 5, "label": ""},  # Generate Token to Return Success
        {"from": 5, "to": 6, "label": ""},  # Return Success to End
        {"from": 4, "to": 6, "label": ""}   # Return Error to End
    ]
    
    # Draw each connection
    for conn in connections:
        from_elem = elements[conn["from"]]
        to_elem = elements[conn["to"]]
        
        # Calculate connection points based on element types and positions
        if from_elem["type"] == "decision":
            if to_elem["x"] < from_elem["x"]:  # Left connection (Yes)
                from_x = from_elem["x"]
                from_y = from_elem["y"] + from_elem["height"]//2
            elif to_elem["x"] > from_elem["x"]:  # Right connection (No)
                from_x = from_elem["x"] + from_elem["width"]
                from_y = from_elem["y"] + from_elem["height"]//2
            else:  # Down connection
                from_x = from_elem["x"] + from_elem["width"]//2
                from_y = from_elem["y"] + from_elem["height"]
        else:
            from_x = from_elem["x"] + from_elem["width"]//2
            from_y = from_elem["y"] + from_elem["height"]
        
        if to_elem["type"] == "decision":
            if to_elem["y"] > from_elem["y"]:  # Connection from above
                to_x = to_elem["x"] + to_elem["width"]//2
                to_y = to_elem["y"]
            else:  # Horizontal connection
                to_y = to_elem["y"] + to_elem["height"]//2
                if from_elem["x"] < to_elem["x"]:  # From left
                    to_x = to_elem["x"]
                else:  # From right
                    to_x = to_elem["x"] + to_elem["width"]
        else:
            to_x = to_elem["x"] + to_elem["width"]//2
            to_y = to_elem["y"]
        
        # Draw line
        if from_elem["type"] == "decision" and (to_elem["x"] < from_elem["x"] or to_elem["x"] > from_elem["x"]):
            # Horizontal then vertical for decision diamond side exits
            mid_y = to_elem["y"] - 20 if to_elem["y"] > from_elem["y"] else to_elem["y"] + to_elem["height"] + 20
            draw.line([(from_x, from_y), (to_x, from_y), (to_x, to_y)], fill='black', width=2)
            
            # Draw label for decision outcomes
            if conn["label"]:
                label_x = (from_x + to_x) // 2
                label_y = from_y - 20 if label_x == from_x else from_y + 5
                draw.text((label_x, label_y), conn["label"], fill='black', font=label_font)
        elif conn["from"] == 4 and conn["to"] == 6:
            # Special case for the "Return Error" to "End" connection
            mid_x = to_x + 80  # Create a bend to the right
            draw.line([(from_x, from_y), (from_x, from_y + 30), (mid_x, from_y + 30), (mid_x, to_y - 30), (to_x, to_y)], fill='black', width=2)
        else:
            # Simple straight line for most connections
            draw.line([(from_x, from_y), (to_x, to_y)], fill='black', width=2)
            
            # Draw label if any
            if conn["label"]:
                mid_x = (from_x + to_x) // 2
                mid_y = (from_y + to_y) // 2 - 10
                draw.text((mid_x, mid_y), conn["label"], fill='black', font=label_font)
    
    # Save the image
    output_path = DIAGRAMS_DIR / "user_login_flowchart.png"
    img.save(output_path)
    print(f"Flowchart diagram saved to {output_path}")
    return output_path

def create_uml_class_diagram():
    """Create a simple UML class diagram."""
    # Create new white image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font_path = BASE_DIR / "fonts" / "DejaVuSans.ttf"
        title_font = ImageFont.truetype(str(font_path), 20)
        label_font = ImageFont.truetype(str(font_path), 16)
        attr_font = ImageFont.truetype(str(font_path), 14)
    except IOError:
        print("Font file not found, using default font")
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        attr_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2 - 100, 20), "User Management UML", fill='black', font=title_font)
    
    # Define classes
    classes = [
        {
            "name": "User",
            "x": 150, "y": 150, "width": 200, "height": 180,
            "attributes": [
                "- id: Integer",
                "- username: String",
                "- email: String",
                "- password: String"
            ],
            "methods": [
                "+ login(): Boolean",
                "+ logout(): void",
                "+ updateProfile(): void"
            ]
        },
        {
            "name": "Customer", 
            "x": 100, "y": 400, "width": 200, "height": 120,
            "attributes": [
                "- customerNumber: String",
                "- loyaltyPoints: Integer"
            ],
            "methods": [
                "+ placeOrder(): Order"
            ]
        },
        {
            "name": "Admin",
            "x": 400, "y": 400, "width": 200, "height": 120,
            "attributes": [
                "- accessLevel: Integer",
                "- department: String"
            ],
            "methods": [
                "+ manageUsers(): void"
            ]
        },
        {
            "name": "Order",
            "x": 550, "y": 150, "width": 200, "height": 160,
            "attributes": [
                "- orderNumber: String",
                "- date: Date",
                "- totalAmount: Decimal"
            ],
            "methods": [
                "+ addItem(): void",
                "+ removeItem(): void",
                "+ calculateTotal(): Decimal"
            ]
        }
    ]
    
    # Draw each class
    for cls in classes:
        # Calculate section heights
        header_height = 30
        attr_height = 20 * len(cls["attributes"])
        method_height = 20 * len(cls["methods"])
        
        # Draw class box
        draw.rectangle(
            [(cls["x"], cls["y"]), 
             (cls["x"] + cls["width"], cls["y"] + cls["height"])],
            fill="lightyellow",
            outline="black",
            width=2
        )
        
        # Draw class name (header section)
        text_width = draw.textlength(cls["name"], font=label_font)
        text_x = cls["x"] + (cls["width"] - text_width) // 2
        draw.text((text_x, cls["y"] + 5), cls["name"], fill='black', font=label_font)
        
        # Draw separator line after class name
        y_line = cls["y"] + header_height
        draw.line([(cls["x"], y_line), (cls["x"] + cls["width"], y_line)], fill='black', width=1)
        
        # Draw attributes
        for i, attr in enumerate(cls["attributes"]):
            draw.text((cls["x"] + 10, y_line + 5 + i * 20), attr, fill='black', font=attr_font)
        
        # Draw separator line after attributes
        y_line = y_line + attr_height
        draw.line([(cls["x"], y_line), (cls["x"] + cls["width"], y_line)], fill='black', width=1)
        
        # Draw methods
        for i, method in enumerate(cls["methods"]):
            draw.text((cls["x"] + 10, y_line + 5 + i * 20), method, fill='black', font=attr_font)
    
    # Draw inheritance relationships
    relationships = [
        {"type": "inheritance", "from": 1, "to": 0},  # Customer inherits from User
        {"type": "inheritance", "from": 2, "to": 0},  # Admin inherits from User
        {"type": "association", "from": 0, "to": 3, "label": "places >"}  # User to Order
    ]
    
    # Draw each relationship
    for rel in relationships:
        from_cls = classes[rel["from"]]
        to_cls = classes[rel["to"]]
        
        if rel["type"] == "inheritance":
            # Calculate connection points
            from_x = from_cls["x"] + from_cls["width"] // 2
            from_y = from_cls["y"]
            to_x = to_cls["x"] + to_cls["width"] // 2
            to_y = to_cls["y"] + to_cls["height"]
            
            # Draw the line
            draw.line([(from_x, from_y), (from_x, from_y - 20), 
                      (to_x, from_y - 20), (to_x, to_y)], fill='black', width=2)
            
            # Draw the inheritance triangle at the top end
            triangle_size = 10
            triangle_points = [
                (to_x, to_y),  # bottom
                (to_x - triangle_size, to_y + triangle_size),  # bottom left
                (to_x + triangle_size, to_y + triangle_size)   # bottom right
            ]
            draw.polygon(triangle_points, fill='white', outline='black')
            
        elif rel["type"] == "association":
            # Calculate connection points
            from_x = from_cls["x"] + from_cls["width"]
            from_y = from_cls["y"] + from_cls["height"] // 3
            to_x = to_cls["x"]
            to_y = to_cls["y"] + to_cls["height"] // 3
            
            # Draw the line
            draw.line([(from_x, from_y), (to_x, to_y)], fill='black', width=2)
            
            # Draw the arrow at the target end
            arrow_size = 10
            # Calculate the angle of the line
            import math
            angle = math.atan2(to_y - from_y, to_x - from_x)
            # Calculate the arrow points
            arrow_angle1 = angle + math.pi * 3/4
            arrow_angle2 = angle - math.pi * 3/4
            arrow_x1 = to_x - arrow_size * math.cos(arrow_angle1)
            arrow_y1 = to_y - arrow_size * math.sin(arrow_angle1)
            arrow_x2 = to_x - arrow_size * math.cos(arrow_angle2)
            arrow_y2 = to_y - arrow_size * math.sin(arrow_angle2)
            
            # Draw the arrow
            draw.line([(to_x, to_y), (arrow_x1, arrow_y1)], fill='black', width=2)
            draw.line([(to_x, to_y), (arrow_x2, arrow_y2)], fill='black', width=2)
            
            # Draw label if provided
            if "label" in rel:
                mid_x = (from_x + to_x) // 2
                mid_y = (from_y + to_y) // 2 - 10
                draw.text((mid_x, mid_y), rel["label"], fill='black', font=attr_font)
    
    # Save the image
    output_path = DIAGRAMS_DIR / "user_management_uml.png"
    img.save(output_path)
    print(f"UML class diagram saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Create the test diagrams
    print("Creating test diagrams for diagram entity extraction...")
    
    arch_diagram = create_architecture_diagram()
    flow_diagram = create_flowchart_diagram()
    uml_diagram = create_uml_class_diagram()
    
    print(f"\nAll test diagrams created successfully in: {DIAGRAMS_DIR}")
    print("Use these diagrams for testing the diagram entity extraction feature.")