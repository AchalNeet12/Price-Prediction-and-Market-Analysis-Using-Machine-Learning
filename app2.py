import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64

# Load the trained model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to set background image and text color
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: white; /* Set all text color to white */
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: white; /* Ensure headers are also white */
    }}
    .stApp .stMarkdown p {{
        color: white; /* Set all paragraph text to white */
    }}
    .stButton button {{
        background-color: #0078D7; /* Optional: Adjust button color */
        color: white;
        border-radius: 5px;
        border: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function to set the background (provide the correct image path)
set_background("img.jpeg")

# Expected columns based on your training dataset
expected_columns = [
    'Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year_2016',
    'year_2017', 'year_2018', 'region_Atlanta', 'region_BaltimoreWashington', 'region_Boise', 'region_Boston',
    'region_BuffaloRochester', 'region_California', 'region_Charlotte', 'region_Chicago', 'region_CincinnatiDayton',
    'region_Columbus', 'region_DallasFtWorth', 'region_Denver', 'region_Detroit', 'region_GrandRapids', 
    'region_GreatLakes', 'region_HarrisburgScranton', 'region_HartfordSpringfield', 'region_Houston', 
    'region_Indianapolis', 'region_Jacksonville', 'region_LasVegas', 'region_LosAngeles', 'region_Louisville', 
    'region_MiamiFtLauderdale', 'region_Midsouth', 'region_Nashville', 'region_NewOrleansMobile', 'region_NewYork', 
    'region_Northeast', 'region_NorthernNewEngland', 'region_Orlando', 'region_Philadelphia', 'region_PhoenixTucson', 
    'region_Pittsburgh', 'region_Plains', 'region_Portland', 'region_RaleighGreensboro', 'region_RichmondNorfolk', 
    'region_Roanoke', 'region_Sacramento', 'region_SanDiego', 'region_SanFrancisco', 'region_Seattle', 
    'region_SouthCarolina', 'region_SouthCentral', 'region_Southeast', 'region_Spokane', 'region_StLouis',
    'region_Syracuse', 'region_Tampa', 'region_TotalUS', 'region_West', 'region_WestTexNewMexico', 'Month_AUG',
    'Month_DEC', 'Month_FEB', 'Month_JAN', 'Month_JULY', 'Month_JUNE', 'Month_MARCH', 'Month_MAY', 'Month_NOV', 
    'Month_OCT', 'Month_SEPT']

# Header for main content
st.title("Avocado Price Prediction 🥑")
st.markdown("Welcome to the **Avocado Price Prediction** App! Use the inputs to predict the price of avocados.")

# Sidebar layout with an enhanced look
st.sidebar.title("📝 Inputs")

# Input fields (inside a stylish sidebar box)
total_volume = st.sidebar.number_input("Total Volume (in lbs)", min_value=0.0, step=0.01)
total_bags = st.sidebar.number_input("Total Bags", min_value=0.0, step=0.01)
small_bags = st.sidebar.number_input("Small Bags", min_value=0.0, step=0.01)
large_bags = st.sidebar.number_input("Large Bags", min_value=0.0, step=0.01)
xlarge_bags = st.sidebar.number_input("XLarge Bags", min_value=0.0, step=0.01)
_4046 = st.sidebar.number_input("4046", min_value=0.0, step=0.01)
_4225 = st.sidebar.number_input("4225", min_value=0.0, step=0.01)
_4770 = st.sidebar.number_input("4770", min_value=0.0, step=0.01)

# Select month, year, and region with dropdowns
month = st.sidebar.selectbox("Select Month", ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
year = st.sidebar.selectbox("Select Year", [2016, 2017, 2018])
region = st.sidebar.selectbox("Select Region", ['Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston', 'BuffaloRochester',
                                               'California', 'Chicago', 'Columbus', 'DallasFtWorth', 'LasVegas', 'LosAngeles', 
                                               'MiamiFtLauderdale', 'NewYork', 'Seattle', 'SanFrancisco'])

# Type selection
type_selected = st.sidebar.selectbox("Select Type", ['conventional', 'organic'])

# One-hot encode the input data (Month, Year, Region, Type)
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
month_encoded = [0] * len(months)
if month in months:
    month_encoded[months.index(month)] = 1

# One-hot encoding for 'Year'
year_encoded = [0, 0, 0]
if year == 2016:
    year_encoded[0] = 1
elif year == 2017:
    year_encoded[1] = 1
elif year == 2018:
    year_encoded[2] = 1

# One-hot encoding for 'Region'
regions = ['Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston', 'BuffaloRochester', 'California', 'Chicago', 
           'Columbus', 'DallasFtWorth', 'LasVegas', 'LosAngeles', 'MiamiFtLauderdale', 'NewYork', 'Seattle', 'SanFrancisco']
region_encoded = [0] * len(regions)
if region in regions:
    region_encoded[regions.index(region)] = 1

# One-hot encoding for 'Type'
type_encoded = [1 if type_selected == 'conventional' else 0]

# Prepare the input data as a DataFrame
input_data = pd.DataFrame([[total_volume, _4046, _4225, _4770, total_bags, small_bags, large_bags, xlarge_bags, *type_encoded, 
                            *year_encoded, *month_encoded, *region_encoded]],
                          columns=expected_columns[:len(month_encoded) + len(year_encoded) + len(region_encoded) + len(type_encoded) + 8])

# Ensure all expected columns are present, add missing ones with zero
missing_columns = set(expected_columns) - set(input_data.columns)
for col in missing_columns:
    input_data[col] = 0

# Reorder columns to match the expected order
input_data = input_data[expected_columns]

# StandardScaler: fit and transform the input data
scaler = StandardScaler()

# Only apply scaling to the features that need it
cols_to_scale = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags','Large Bags', 'XLarge Bags']

# Apply fit_transform on input_data (fit and scale on the input)
input_data_scaled = input_data.copy()
input_data_scaled[cols_to_scale] = scaler.fit_transform(input_data[cols_to_scale])

# Prediction
if st.sidebar.button("📈 Predict Price"):
    prediction = model.predict(input_data_scaled)
    st.subheader(f"**Predicted Price: ${prediction[0]:.2f}**")

    # Display a nice message based on prediction
    if prediction[0] > 2:
        st.success(f"The predicted price is **${prediction[0]:.2f}**. Looks like it's going to be a great time to buy!")
    elif prediction[0] < 1.5:
        st.warning(f"The predicted price is **${prediction[0]:.2f}**. Consider waiting a bit for a better deal.")
    else:
        st.info(f"The predicted price is **${prediction[0]:.2f}**. It's a good time to buy!")
