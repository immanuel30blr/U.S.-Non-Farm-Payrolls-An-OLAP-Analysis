import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime
import os

# --- Helper Functions ---

def add_custom_css():
    """Injects custom CSS for a cleaner look."""
    st.markdown(
        """
        <style>
        .css-1d391kg {
            padding-top: 3.5rem;
            padding-bottom: 3.5rem;
        }
        .main-header {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 20px;
        }
        .st-emotion-cache-18ni4h1 {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .st-emotion-cache-z5304v {
            background-color: #f7f9fc;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .st-emotion-cache-1629p8f {
            color: #007bff;
            font-weight: 600;
        }
        .streamlit-expanderHeader {
            font-size: 1.1em;
            font-weight: bold;
            color: #4a4a4a;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data
def load_data():
    """
    Connects to PostgreSQL, loads data, and caches it.
    Returns: DataFrame on success, None on failure.
    """
    conn = None
    try:
        # Using hardcoded values as per user's request to bypass env variables
        conn = psycopg2.connect(
            dbname="ETL_NFP_G1",
            user="postgres",
            password="bijujohn",
            host="localhost",
            port="5432"
        )
        st.success("✅ Successfully connected to the database!")
        query = "SELECT * FROM nonfarm_payrolls;"
        df = pd.read_sql(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        return df

    except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
        st.error(f"❌ Error connecting to the database or loading data: {e}")
        return None

def create_slicing_charts(df):
    """
    Performs Slicing operations and creates visualizations.
    - Slices by year and by specific months.
    """
    st.subheader("Slicing Analysis")

    # Question 1: Average total payroll employment for each calendar year (2010–2025).
    st.info("📊 **Question 1:** Average total payroll employment by year.")
    df_avg_jobs = df.groupby(df['date'].dt.year)['total_payroll_employment'].mean().reset_index()
    df_avg_jobs.columns = ['year', 'avg_employment']

    # SQL behind the scenes
    with st.expander("Show the underlying SQL"):
        st.code("""
        SELECT
            EXTRACT(YEAR FROM date) AS year,
            AVG(total_payroll_employment) AS avg_employment
        FROM nonfarm_payrolls
        GROUP BY year
        ORDER BY year;
        """)

    fig1 = px.bar(
        df_avg_jobs,
        x='year',
        y='avg_employment',
        title='Average Total Payroll Employment by Year',
        labels={'year': 'Year', 'avg_employment': 'Average Employment'},
        color='avg_employment',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Question 2: Monthly employment levels from March–December 2020 vs. 2019.
    st.info("📊 **Question 2:** Monthly employment levels for March-Dec, 2020 vs 2019.")
    df_compare = df[
        (df['date'].dt.year.isin([2019, 2020])) &
        (df['date'].dt.month.isin(range(3, 13)))
    ]
    df_compare['year'] = df_compare['date'].dt.year.astype(str)
    
    # SQL behind the scenes
    with st.expander("Show the underlying SQL"):
        st.code("""
        SELECT
            date,
            total_payroll_employment,
            EXTRACT(YEAR FROM date) AS year
        FROM nonfarm_payrolls
        WHERE
            EXTRACT(YEAR FROM date) IN (2019, 2020) AND
            EXTRACT(MONTH FROM date) BETWEEN 3 AND 12
        ORDER BY date;
        """)

    fig2 = px.line(
        df_compare,
        x='date',
        y='total_payroll_employment',
        color='year',
        title='Monthly Employment: Mar-Dec 2020 vs 2019',
        labels={'date': 'Date', 'total_payroll_employment': 'Total Employment', 'year': 'Year'},
        hover_data={'total_payroll_employment': True, 'year': False}
    )
    st.plotly_chart(fig2, use_container_width=True)

def create_dicing_charts(df):
    """
    Performs Dicing operations and creates visualizations.
    - Filters by conditions to identify specific events.
    """
    st.subheader("Dicing Analysis")

    # Question 1: Months with >2% MoM drop & recovery time.
    st.info("📊 **Question 1:** Months with >2% MoM employment drop and recovery time.")
    df['mom_pct_change'] = df['total_payroll_employment'].pct_change() * 100
    df_drops = df[df['mom_pct_change'] < -2].copy()

    # Calculate recovery time for each drop
    for index, row in df_drops.iterrows():
        start_date = row['date']
        start_employment = row['total_payroll_employment']
        
        # Find the previous peak to measure recovery
        df_prior_peak = df[df['date'] < start_date]['total_payroll_employment'].max()
        
        # Find when employment recovered to the prior peak
        df_recovery = df[(df['date'] > start_date) & (df['total_payroll_employment'] >= df_prior_peak)]
        
        if not df_recovery.empty:
            recovery_date = df_recovery['date'].min()
            months_to_recover = (recovery_date.year - start_date.year) * 12 + (recovery_date.month - start_date.month)
            st.markdown(f"**Drop identified in:** :blue[{start_date.strftime('%B %Y')}]")
            st.write(f"Prior peak employment: {df_prior_peak:,.0f} (reached on {df[df['total_payroll_employment'] == df_prior_peak]['date'].iloc[0].strftime('%B %Y')})")
            st.write(f"It took **{months_to_recover} months** to recover to that peak.")

    # SQL behind the scenes
    with st.expander("Show the underlying SQL Logic"):
        st.markdown(
            """
            This analysis requires a multi-step process that's best handled in a programming language like Python, as SQL doesn't have a direct way to calculate dynamic "time to recovery" based on prior peaks for each drop. However, the core dicing operation would look like this:
            ```sql
            WITH monthly_change AS (
              SELECT
                date,
                total_payroll_employment,
                (total_payroll_employment - LAG(total_payroll_employment) OVER (ORDER BY date)) * 100.0 / LAG(total_payroll_employment) OVER (ORDER BY date) AS mom_pct_change
              FROM nonfarm_payrolls
            )
            SELECT *
            FROM monthly_change
            WHERE mom_pct_change < -2.0;
            ```
            """
        )

    # Question 2: Which month consistently shows the highest payroll growth in Q4?
    st.info("📊 **Question 2:** Within Q4, which month consistently shows the highest payroll growth?")
    df_q4 = df[df['date'].dt.month.isin([10, 11, 12])].copy()
    df_q4['mom_pct_change'] = df_q4.groupby(df_q4['date'].dt.year)['total_payroll_employment'].pct_change() * 100
    df_q4 = df_q4.dropna()
    df_q4['month'] = df_q4['date'].dt.strftime('%B')
    
    # SQL behind the scenes
    with st.expander("Show the underlying SQL Logic"):
        st.markdown(
            """
            This is an aggregation and comparison query.
            ```sql
            WITH mom_change AS (
                SELECT
                    date,
                    total_payroll_employment,
                    (total_payroll_employment - LAG(total_payroll_employment) OVER (ORDER BY date)) * 100.0 / LAG(total_payroll_employment) OVER (ORDER BY date) AS mom_pct_change
                FROM nonfarm_payrolls
            )
            SELECT
                EXTRACT(MONTH FROM date) AS month,
                AVG(mom_pct_change) AS avg_q4_growth
            FROM mom_change
            WHERE
                EXTRACT(MONTH FROM date) IN (10, 11, 12)
            GROUP BY month
            ORDER BY avg_q4_growth DESC;
            ```
            """
        )

    df_q4_avg = df_q4.groupby('month')['mom_pct_change'].mean().reset_index()
    fig_q4 = px.bar(
        df_q4_avg,
        x='month',
        y='mom_pct_change',
        title='Average Q4 MoM Growth by Month',
        labels={'month': 'Month', 'mom_pct_change': 'Average MoM Growth (%)'}
    )
    st.plotly_chart(fig_q4, use_container_width=True)

def create_rollup_charts(df):
    """
    Performs Roll-up operations and creates visualizations.
    - Aggregates data to higher levels (e.g., quarter, decade).
    """
    st.subheader("Roll-up Analysis")

    # Question 1: Aggregate by quarter and year with growth rates.
    st.info("📊 **Question 1:** Aggregated employment by quarter and year with growth rates.")
    df_agg = df.copy()
    df_agg['year'] = df_agg['date'].dt.year
    df_agg['quarter'] = df_agg['date'].dt.to_period('Q')
    df_quarterly = df_agg.groupby('quarter')['total_payroll_employment'].sum().reset_index()
    df_quarterly['quarter_over_quarter'] = df_quarterly['total_payroll_employment'].pct_change() * 100

    df_yearly = df_agg.groupby('year')['total_payroll_employment'].sum().reset_index()
    df_yearly['year_over_year'] = df_yearly['total_payroll_employment'].pct_change() * 100

    # SQL behind the scenes
    with st.expander("Show the underlying SQL Logic"):
        st.markdown(
            """
            This is a two-step aggregation process.
            ```sql
            -- Quarterly Roll-up with QoQ Growth
            WITH quarterly_data AS (
                SELECT
                    EXTRACT(YEAR FROM date) AS year,
                    EXTRACT(QUARTER FROM date) AS quarter,
                    SUM(total_payroll_employment) AS total_employment
                FROM nonfarm_payrolls
                GROUP BY 1, 2
                ORDER BY 1, 2
            )
            SELECT
                year,
                quarter,
                total_employment,
                (total_employment - LAG(total_employment, 1) OVER (ORDER BY year, quarter)) * 100.0 / LAG(total_employment, 1) OVER (ORDER BY year, quarter) AS qoq_growth_rate
            FROM quarterly_data;

            -- Yearly Roll-up with YoY Growth
            WITH yearly_data AS (
                SELECT
                    EXTRACT(YEAR FROM date) AS year,
                    SUM(total_payroll_employment) AS total_employment
                FROM nonfarm_payrolls
                GROUP BY 1
                ORDER BY 1
            )
            SELECT
                year,
                total_employment,
                (total_employment - LAG(total_employment, 1) OVER (ORDER BY year)) * 100.0 / LAG(total_employment, 1) OVER (ORDER BY year) AS yoy_growth_rate
            FROM yearly_data;
            ```
            """
        )

    st.write("Quarterly and Annual Employment Growth Rates")
    st.dataframe(df_quarterly.head(5).style.format({'total_payroll_employment': '{:,.0f}', 'quarter_over_quarter': '{:,.2f}%'}), hide_index=True)
    st.dataframe(df_yearly.style.format({'total_payroll_employment': '{:,.0f}', 'year_over_year': '{:,.2f}%'}), hide_index=True)

    # Question 2: Compare average employment in the 2010s vs. the 2000s.
    st.info("📊 **Question 2:** Comparing average employment in the 2010s vs. the 2000s.")
    df['decade'] = df['date'].dt.year.apply(lambda y: f"{int(y // 10) * 10}s")
    df_decade_avg = df.groupby('decade')['total_payroll_employment'].mean().reset_index()

    # SQL behind the scenes
    with st.expander("Show the underlying SQL"):
        st.code("""
        SELECT
            CASE
                WHEN EXTRACT(YEAR FROM date) BETWEEN 2000 AND 2009 THEN '2000s'
                WHEN EXTRACT(YEAR FROM date) BETWEEN 2010 AND 2019 THEN '2010s'
                ELSE 'Other'
            END AS decade,
            AVG(total_payroll_employment) AS avg_employment
        FROM nonfarm_payrolls
        GROUP BY decade
        ORDER BY decade;
        """)

    fig_decade = px.bar(
        df_decade_avg,
        x='decade',
        y='total_payroll_employment',
        title='Average Employment by Decade',
        labels={'total_payroll_employment': 'Average Employment', 'decade': 'Decade'}
    )
    st.plotly_chart(fig_decade, use_container_width=True)

def create_drilldown_charts(df):
    """
    Performs Drill-down operations and creates visualizations.
    - Breaks down high-level data into granular details.
    """
    st.subheader("Drill-Down Analysis")

    # Question 1: For the year with the highest annual employment gain, show which months contributed most.
    st.info("📊 **Question 1:** The year with the highest employment gain and its monthly breakdown.")
    
    # Calculate annual employment gain
    df_annual = df.groupby(df['date'].dt.year)['total_payroll_employment'].sum().reset_index()
    df_annual['annual_gain'] = df_annual['total_payroll_employment'].diff()
    top_gain_year = df_annual.loc[df_annual['annual_gain'].idxmax()]
    top_gain_year_value = top_gain_year['annual_gain']
    top_gain_year_label = int(top_gain_year['date'])
    
    st.write(f"The year with the highest annual employment gain was **{top_gain_year_label}**, with a gain of **{top_gain_year_value:,.0f}** jobs.")

    # Drill down into that year's monthly data
    df_top_year = df[df['date'].dt.year == top_gain_year_label].copy()
    df_top_year['month'] = df_top_year['date'].dt.strftime('%B')
    
    # SQL behind the scenes
    with st.expander("Show the underlying SQL Logic"):
        st.markdown(
            f"""
            To find the year with the highest gain, you'd use a window function. Then you would perform a second query to get the monthly data for that specific year:
            ```sql
            -- Step 1: Find the year with the max annual gain
            WITH yearly_employment AS (
                SELECT
                    EXTRACT(YEAR FROM date) AS year,
                    SUM(total_payroll_employment) AS total_employment
                FROM nonfarm_payrolls
                GROUP BY year
                ORDER BY year
            ),
            yearly_gain AS (
                SELECT
                    year,
                    total_employment - LAG(total_employment, 1) OVER (ORDER BY year) AS annual_gain
                FROM yearly_employment
            )
            SELECT year
            FROM yearly_gain
            ORDER BY annual_gain DESC
            LIMIT 1;

            -- Step 2: Drill down into the winning year (e.g., 2020)
            SELECT date, total_payroll_employment
            FROM nonfarm_payrolls
            WHERE EXTRACT(YEAR FROM date) = {top_gain_year_label}
            ORDER BY date;
            ```
            """
        )

    fig_drill_1 = px.bar(
        df_top_year,
        x='month',
        y='total_payroll_employment',
        title=f"Monthly Employment in {top_gain_year_label}",
        labels={'total_payroll_employment': 'Total Employment', 'month': 'Month'},
        color='total_payroll_employment',
        color_continuous_scale=px.colors.sequential.Plotly3
    )
    st.plotly_chart(fig_drill_1, use_container_width=True)

    # Question 2: Identify the month with the sharpest drop and explain.
    st.info("📊 **Question 2:** The month with the sharpest employment drop.")
    df['mom_gain'] = df['total_payroll_employment'].pct_change() * 100
    sharpest_drop_month = df.loc[df['mom_gain'].idxmin()]
    
    st.write(f"The sharpest drop was in **{sharpest_drop_month['date'].strftime('%B %Y')}**.")
    st.write(f"The employment level dropped by **{sharpest_drop_month['mom_gain']:.2f}%** that month.")
    st.warning("Note: This dataset is monthly, so we cannot drill down further to a weekly breakdown.")
    
    # SQL behind the scenes
    with st.expander("Show the underlying SQL Logic"):
        st.code("""
        SELECT
            date,
            total_payroll_employment,
            (total_payroll_employment - LAG(total_payroll_employment, 1) OVER (ORDER BY date)) * 100.0 / LAG(total_payroll_employment, 1) OVER (ORDER BY date) AS mom_growth
        FROM nonfarm_payrolls
        ORDER BY mom_growth ASC
        LIMIT 1;
        """)


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="NFP OLAP Dashboard")
    add_custom_css()
    st.markdown("<h1 class='main-header'>U.S. Non-Farm Payrolls: An OLAP Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.header("Navigation")

    # Load data once and cache it
    data = load_data()
    if data is None:
        st.stop() # Stop the app if data loading failed

    # Sidebar for selecting the OLAP operation
    operation = st.sidebar.radio(
        "Choose an OLAP Operation",
        ('Slicing', 'Dicing', 'Roll-up', 'Drill-Down')
    )

    if operation == 'Slicing':
        create_slicing_charts(data.copy())
    elif operation == 'Dicing':
        create_dicing_charts(data.copy())
    elif operation == 'Roll-up':
        create_rollup_charts(data.copy())
    elif operation == 'Drill-Down':
        create_drilldown_charts(data.copy())

if __name__ == "__main__":
    main()
