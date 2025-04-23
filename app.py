import streamlit as st
import pickle
import numpy as np

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Streamlit config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Predictor")
st.write("Fill in the specs below to estimate the laptop's price.")

# Inputs (organized neatly)
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    Type = st.selectbox('Laptop Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

with col2:
    ips = st.selectbox('IPS Display', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)
    resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                             '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())

# Storage and OS (below)
st.markdown("### Storage & OS")
col3, col4 = st.columns(2)

with col3:
    HDD = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
with col4:
    SSD = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024, 2048])

os = st.selectbox('Operating System', df['os'].unique())

# Predict button
if st.button('Predict Price'):
    # Convert inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))

    if screen_size > 0:
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size
    else:
        st.error("Screen size must be greater than 0.")
        st.stop()

    # Feature engineering (match training preprocessing)
    cpu_brand = cpu.split()[0] if cpu else "Unknown"
    gpu_brand = gpu.split()[0] if gpu else "Unknown"

    # Construct query as DataFrame
    import pandas as pd
    query_dict = {
        'Company': [company],
        'TypeName': [Type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu_brand],
        'HDD': [HDD],
        'SSD': [SSD],
        'Gpu brand': [gpu_brand],
        'os': [os]
    }

    query_df = pd.DataFrame(query_dict)

    # Ensure all string columns are proper strings
    for col in query_df.select_dtypes(include=['object']).columns:
        query_df[col] = query_df[col].astype(str)

    # Predict using the pipeline
    predicted_price = int(np.exp(pipe.predict(query_df)[0]))
    st.success(f"💰 Estimated Price: ₹{predicted_price:,}")
