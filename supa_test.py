import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
# Make sure you have a .env file in the same directory as this script
load_dotenv()

# --- Environment Variable Names ---
# These are the names your variables should have inside your .env file
ENV_VAR_USER = "DB_POOLER_USER"
ENV_VAR_PASSWORD = "DB_PASSWORD"
ENV_VAR_HOST = "DB_POOLER_HOST"
ENV_VAR_PORT = "DB_POOLER_PORT"
ENV_VAR_DBNAME = "DB_NAME"

# --- Fetch variables from environment ---
USER = os.getenv(ENV_VAR_USER)
PASSWORD = os.getenv(ENV_VAR_PASSWORD)
HOST = os.getenv(ENV_VAR_HOST)
PORT = os.getenv(ENV_VAR_PORT)
DBNAME = os.getenv(ENV_VAR_DBNAME)

# --- Check if variables were loaded ---
missing_vars = []
if not USER: missing_vars.append(ENV_VAR_USER)
if not PASSWORD: missing_vars.append(ENV_VAR_PASSWORD)
if not HOST: missing_vars.append(ENV_VAR_HOST)
if not PORT: missing_vars.append(ENV_VAR_PORT)
if not DBNAME: missing_vars.append(ENV_VAR_DBNAME)

if missing_vars:
    print(f"Error: Required environment variable(s) not found in .env file: {', '.join(missing_vars)}")
    print("Please ensure your .env file is set up correctly.")
    exit() # Stop the script if config is missing

# --- Connect to the database ---
try:
    print(f"Attempting to connect to host '{HOST}' on port '{PORT}' as user '{USER}'...")
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )
    print("Connection successful!")

    # Create a cursor to execute SQL queries
    cursor = connection.cursor()

    # Example query
    print("Executing query: SELECT NOW();")
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    print("Current Time:", result)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    print("Connection closed.")

except Exception as e:
    # Print the specific error encountered
    print(f"Failed to connect or execute query: {e}")