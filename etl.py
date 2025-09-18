import pandas as pd
import psycopg2
from fredapi import Fred
import os

# --- Database Connection Details ---
# Hardcoding these values to ensure they are used,
# bypassing any environment variables that might be set.
DB_NAME = "ETL_NFP_G1"
DB_USER = "postgres"
DB_PASSWORD = "bijujohn"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Environment Variable for FRED API Key ---
# You MUST replace the placeholder below with your actual FRED API key.
FRED_API_KEY = os.getenv("FRED_API_KEY", "57bba6956650af0ae0ba657b29a5608b")

def extract_data():
    """Extracts non-farm payrolls data from the FRED API."""
    print("Step 1: Extracting data from FRED...")
    fred = Fred(api_key=FRED_API_KEY)
    # Series ID for Total Non-Farm Payrolls
    df_raw = fred.get_series('PAYEMS', observation_start='2019-01-01')
    return df_raw.reset_index()

def transform_data(df):
    """
    Transforms the raw data:
    - Renames columns for clarity.
    - Calculates month-over-month change in employment.
    """
    print("Step 2: Transforming data (calculating MoM change)...")
    df.columns = ['date', 'total_payroll_employment']
    df['total_payroll_employment'] = df['total_payroll_employment'].astype(int)
    # For this specific app, MoM change isn't used in the dashboard but is a common ETL step.
    df['mom_change'] = df['total_payroll_employment'].pct_change() * 100
    df = df.dropna()
    return df

def load_data(df):
    """
    Loads the transformed data into the PostgreSQL database.
    """
    conn = None  # Initialize conn to None to prevent NameError
    try:
        print("Step 3: Loading transformed data into PostgreSQL...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        # Create table if it doesn't exist (this is an idempotent operation)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nonfarm_payrolls (
                date DATE PRIMARY KEY,
                total_payroll_employment INT
            );
        """)
        
        # Load data row by row (simple but robust for this size)
        for index, row in df.iterrows():
            cur.execute("""
                INSERT INTO nonfarm_payrolls (date, total_payroll_employment)
                VALUES (%s, %s)
                ON CONFLICT (date) DO UPDATE SET total_payroll_employment = EXCLUDED.total_payroll_employment;
            """, (row['date'], row['total_payroll_employment']))
            
        conn.commit()
        print("ETL pipeline completed successfully.")
        
    except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
        # Now this block will catch the auth error and print a helpful message
        print(f"An error occurred: {e}")
        print("Please check your database credentials and ensure the database is running.")
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # This is the main ETL pipeline execution
    df_raw = extract_data()
    df_transformed = transform_data(df_raw)
    load_data(df_transformed)
