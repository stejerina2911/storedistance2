import streamlit as st
import pandas as pd
import numpy as np
import io
from geopy.distance import geodesic

def main():
    st.title("Store Distance Calculator")
    st.write("""
        Welcome to the **Store Distance Calculator**. This tool allows you to calculate the minimum distance between your stores and the nearest competitor stores.
        Please follow the instructions below to upload your file and process the data.
    """)

    st.header("Instructions")
    st.markdown("""
    **Steps to Use the Application:**

    1. **Prepare Your Excel File**: Ensure your Excel file contains the following columns:
       - `Banner`: A column indicating whether a store is 'DK' (your store) or a competitor.
       - `LAT`: The latitude of the store.
       - `LONG`: The longitude of the store.
    2. **Upload the File**: Use the file uploader below to upload your Excel file.
    3. **Process the File**: Click on the **Process File** button to calculate the distances.
    4. **Download the Updated File**: Once processing is complete, a download button will appear to download the updated Excel file.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            # Check if necessary columns are present
            required_columns = {'Banner', 'LAT', 'LONG'}
            if not required_columns.issubset(df.columns):
                st.error(f"The uploaded file must contain the following columns: {required_columns}")
                return

            st.success("File uploaded successfully!")
            st.write("Here's a preview of your data:")
            st.dataframe(df.head())

            if st.button('Process File'):
                with st.spinner('Calculating distances...'):
                    processed_df = calculate_distances(df)

                st.success('Processing complete!')
                st.write("Here's a preview of the processed data:")
                st.dataframe(processed_df.head())

                # Create a download button for the updated file
                towrite = io.BytesIO()
                processed_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button(
                    label="Download Updated File",
                    data=towrite,
                    file_name="updated_stores.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    This function is vectorized for efficiency.
    """
    # Convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute differences
    dlat = lat2[np.newaxis, :] - lat1[:, np.newaxis]
    dlon = lon2[np.newaxis, :] - lon1[:, np.newaxis]

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1)[:, np.newaxis] * np.cos(lat2)[np.newaxis, :] * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in kilometers (mean radius)
    km = 6371 * c
    return km

def calculate_distances(df):
    # Separate 'DK' stores and competitor stores
    dk_stores = df[df['Banner'] == 'DK'].reset_index()
    competitor_stores = df[df['Banner'] != 'DK'].reset_index()

    if dk_stores.empty or competitor_stores.empty:
        st.error("Either 'DK' stores or competitor stores are missing in the data.")
        return df

    # Create new columns for storing the minimum distances
    df['Min Competitor Distance (km)'] = None
    df['Min DK Distance (km)'] = None

    # Calculate distances from DK stores to competitor stores
    lat1 = dk_stores['LAT'].values
    lon1 = dk_stores['LONG'].values
    lat2 = competitor_stores['LAT'].values
    lon2 = competitor_stores['LONG'].values

    distances = haversine_array(lat1, lon1, lat2, lon2)
    min_distances = distances.min(axis=1)
    dk_stores['Min Competitor Distance (km)'] = min_distances

    # Update original dataframe with the new distances for DK stores
    for idx, row in dk_stores.iterrows():
        df.at[row['index'], 'Min Competitor Distance (km)'] = row['Min Competitor Distance (km)']

    # Calculate distances from competitor stores to DK stores
    lat1 = competitor_stores['LAT'].values
    lon1 = competitor_stores['LONG'].values
    lat2 = dk_stores['LAT'].values
    lon2 = dk_stores['LONG'].values

    distances = haversine_array(lat1, lon1, lat2, lon2)
    min_distances = distances.min(axis=1)
    competitor_stores['Min DK Distance (km)'] = min_distances

    # Update original dataframe with the new distances for competitor stores
    for idx, row in competitor_stores.iterrows():
        df.at[row['index'], 'Min DK Distance (km)'] = row['Min DK Distance (km)']

    return df

if __name__ == '__main__':
    main()
