# Test Diagrams for Entity Extraction

This directory contains test diagrams for evaluating the diagram entity extraction feature. The diagrams are created using the `create_test_diagrams.py` script and are designed to test the system's ability to extract entities and relationships from different diagram types.

## Available Diagrams

1. **System Architecture Diagram** - `system_architecture.png`
   - Entity types: Component, Service, Database
   - Relationship types: HTTP connection, REST API, SQL query

2. **User Login Flowchart** - `user_login_flowchart.png`
   - Entity types: Process, Decision
   - Relationship types: Sequence, Conditional flow

3. **User Management UML** - `user_management_uml.png`
   - Entity types: Class, Attribute, Method
   - Relationship types: Inheritance, Association

## Generating the Diagrams

To generate the test diagrams, run the `create_test_diagrams.py` script:

```bash
cd /path/to/LightRAG
python tests/fixtures/create_test_diagrams.py
```

This will create the diagram PNG files in this directory.

## Using the Test Diagrams

These diagrams can be used for:

1. **Unit Testing**: Test the `DiagramEntityExtractor` class with these diagrams to verify it correctly extracts entities and relationships.

2. **Integration Testing**: Test the end-to-end pipeline from diagram extraction to knowledge graph integration.

3. **Evaluation**: Use these diagrams as part of your evaluation dataset to measure the quality of entity and relationship extraction.

## Expected Entities and Relationships

### System Architecture Diagram

**Expected Entities:**
- Frontend (Component)
- API Gateway (Service)
- User Service (Service)
- Content Service (Service)
- Database (DataStore)

**Expected Relationships:**
- Frontend CALLS API Gateway (HTTP)
- API Gateway CALLS User Service (REST)
- API Gateway CALLS Content Service (REST)
- User Service USES Database (SQL)
- Content Service USES Database (SQL)

### User Login Flowchart

**Expected Entities:**
- User Login Request (Process)
- Valid Credentials? (Decision)
- Generate Token (Process)
- Return Error (Process)
- Return Success (Process)

**Expected Relationships:**
- User Login Request NEXT Valid Credentials?
- Valid Credentials? NEXT_IF Generate Token (Yes)
- Valid Credentials? NEXT_IF Return Error (No)
- Generate Token NEXT Return Success
- Return Success NEXT End
- Return Error NEXT End

### User Management UML

**Expected Entities:**
- User (Class)
- Customer (Class)
- Admin (Class)
- Order (Class)

**Expected Relationships:**
- Customer INHERITS_FROM User
- Admin INHERITS_FROM User
- User ASSOCIATES_WITH Order (places)

## Customizing the Diagrams

To create new test diagrams or customize the existing ones, modify the `create_test_diagrams.py` script. The script uses the PIL library to create simple diagrams and can be extended to generate more complex test cases.