import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# Set default values
default_reaction_type = "dislike"
default_start_date = datetime.now() - timedelta(days=30)
default_end_date = datetime.now()

# Streamlit UI
st.title("Reaction Monitoring Dashboard")

reaction_type = st.selectbox(
    "Select Reaction Type:",
    options=["like", "dislike", "regenerate"],
    index=1  # default is "dislike"
)

start_date = st.date_input(
    "Select Start Date:",
    value=default_start_date.date()
)

end_date = st.date_input(
    "Select End Date:",
    value=default_end_date.date()
)

plot_type = st.selectbox(
    "Select Plot Type:",
    options = ['Scatter', "Line"]
)

# Ensure that end_date is after start_date
if start_date > end_date:
    st.error("End Date must be after Start Date.")
    st.stop()

# Convert dates to string format with time component
start_date_str = datetime.combine(start_date, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
end_date_str = datetime.combine(end_date, datetime.max.time()).strftime("%Y-%m-%d %H:%M:%S")

# Function to fetch reaction data from the FastAPI backend
def fetch_reaction_data(reaction_type, start_date, end_date):
    url = f"http://localhost:8000/all-reactions"
    params = {
        "reaction_type": reaction_type,
        "start_datetime": start_date,
        "end_datetime": end_date
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data['reactions']
    else:
        st.error("Failed to fetch data from the API.")
        return None

# Fetch the data from FastAPI
reactions = fetch_reaction_data(reaction_type, start_date_str, end_date_str)

if reactions is not None and len(reactions) > 0:
    # Convert the list of reactions into a DataFrame
    df = pd.DataFrame(reactions)
    
    # Convert 'created_at' to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Group reactions by date
    df_grouped = df.groupby(df['created_at'].dt.date).size().reset_index(name='count')
    
    # Rename columns for clarity
    df_grouped.rename(columns={'created_at': 'Date'}, inplace=True)
    
    # Set 'Date' as datetime
    df_grouped['Date'] = pd.to_datetime(df_grouped['Date'])
    
    # Sort by date
    df_grouped.sort_values('Date', inplace=True)
    
    # Set 'Date' as index
    df_grouped.set_index('Date', inplace=True)

    print(df_grouped['count'])
    
    # Show the line chart
    if plot_type == "Scatter":
        st.scatter_chart(df_grouped['count'])
    else:
        st.line_chart(df_grouped['count'])
else:
    st.warning("No reactions found for the selected criteria.")