# import streamlit as st
# import pandas as pd
# import re
# from datetime import datetime
# import io

# def process_excel_file(df_uploaded, file_name):
#     """
#     Processes a single pandas DataFrame (from an uploaded Excel file)
#     according to the logic provided in the notebook code.
#     Returns the processed DataFrame and any error messages.
#     """
#     compiled_results = []
#     error_logs = []

#     # ==== Step 1: Prepare Identifiers ====
#     # (A) Only text match needed
#     simple_identifiers = [
#         'no load',
#         'Male rotor speed',
#         'Nominal motor rating',
#         'Drive motor speed',
#         'Fan motor rating'
#     ]
#     simple_identifiers = [id_.strip().lower() for id_ in simple_identifiers]

#     # (B) Text + Unit match needed
#     unit_identifiers = [
#         ('5 bar', 'm3/min'), ('5 bar', 'KW'), ('6 bar', 'm3/min'), ('6 bar', 'KW'),
#         ('7 bar', 'm3/min'), ('7 bar', 'KW'), ('7.5 bar', 'm3/min'), ('7.5 bar', 'KW'),
#         ('8 bar', 'm3/min'), ('8 bar', 'KW'), ('9 bar', 'm3/min'), ('9 bar', 'KW'),
#         ('10 bar', 'm3/min'), ('10 bar', 'KW'), ('11 bar', 'm3/min'), ('11 bar', 'KW'),
#         ('', 'kW'), # Note: This is for a standalone 'kW' unit
#         ('Performance Data', 'Performance Data'), # Special case for header
#         ('Performance Data', 'bar g'), # Special case for header
#         ('Maximum working pressure', 'bar g') # Special case for header
#     ]
#     unit_identifiers = [(i[0].strip().lower(), i[1].strip().lower()) for i in unit_identifiers]

#     try:
#         df = df_uploaded.copy() # Work on a copy of the uploaded DataFrame
        
#         # Initial check for empty or too few rows
#         if df.empty or len(df) < 5:
#             raise ValueError("Empty or too few rows in the uploaded file. Minimum 5 rows required.")

#         # === 1. Cooling Media detection FIRST ===
#         # Convert top 15 rows to string and lowercase for robust searching
#         top_rows_raw = df.head(15).astype(str).apply(lambda x: x.str.lower())

#         cooling_media = 'Unknown'
#         if top_rows_raw.apply(lambda x: x.str.contains('water', na=False).any()).any():
#             cooling_media = 'W'
#         elif top_rows_raw.apply(lambda x: x.str.contains('air', na=False).any()).any():
#             cooling_media = 'A'

#         # Detect Hz value (50 or 60 Hz)
#         hz_value = None
#         # Iterate through all cell values in the top rows
#         for cell_value in top_rows_raw.values.flatten():
#             match = re.search(r'\b(50|60)\s*hz\b', str(cell_value), re.IGNORECASE)
#             if match:
#                 hz_value = int(match.group(1)) # Extract 50 or 60
#                 break

#         # Drop columns where the value in the 5th row (index 4) is NaN
#         # This assumes the 5th row (index 4) is crucial for identifying valid data columns
#         cols_to_drop = [col for col in df.columns[1:] if pd.isna(df.loc[4, col])]
#         df_cleaned = df.drop(columns=cols_to_drop)

#         # Detect date row using multiple patterns
#         date_patterns = [
#             r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}\.\d{2}\.\d{4}',
#             r'\d{4}/\d{2}/\d{2}', r'\b\d{1,2}[ ]?[A-Za-z]{3,9}[ ]?\d{4}\b',
#             r'\b[A-Za-z]{3,9}[ ]?\d{1,2},?[ ]?\d{4}\b'
#         ]
#         combined_pattern = '|'.join(date_patterns)
#         # Check each row if any cell contains a date pattern
#         date_row_mask = df_cleaned.apply(lambda row: row.astype(str).str.contains(combined_pattern, regex=True, case=False, na=False).any(), axis=1)

#         if not date_row_mask.any():
#             raise ValueError("No date row found in the uploaded file. Please ensure a date is present in a row.")
#         date_row_index = df_cleaned[date_row_mask].index[0] # Get the index of the first row containing a date

#         original_row = df_cleaned.loc[date_row_index].copy()
#         # Create a 'cleaned' version of the date row, masking out the date string itself
#         cleaned_row = original_row.mask(original_row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}'))
#         # Concatenate this 'cleaned' row back to the DataFrame.
#         # This part of the original notebook code is a bit unusual but maintained for consistency.
#         df_cleaned = pd.concat([df_cleaned, pd.DataFrame([cleaned_row], columns=df_cleaned.columns)], ignore_index=True)

#         # Extract actual date objects from the original date row
#         date_strings = original_row[original_row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}')].astype(str)
#         date_objects = pd.to_datetime(date_strings, errors='coerce').dropna().tolist()

#         # Forward fill specific rows for data propagation
#         # Ensure indices exist before attempting to fill
#         rows_to_fill_indices = [idx for idx in [3, 5, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, df_cleaned.index[-1]] if idx in df_cleaned.index]
#         if rows_to_fill_indices:
#             df_cleaned.loc[rows_to_fill_indices] = df_cleaned.loc[rows_to_fill_indices].ffill(axis=1)
#         # Forward fill the first column (index 0)
#         df_cleaned.iloc[:, 0] = df_cleaned.iloc[:, 0].ffill()

#         # Build new header by concatenating values from row 3 and row 4
#         # This assumes row 3 and 4 contain parts of the desired column names
#         new_row_for_header = (df_cleaned.iloc[3].astype(str) + '_' + df_cleaned.iloc[4].astype(str))
#         # Add this new header row to the end of the DataFrame temporarily
#         df_cleaned.loc[df_cleaned.index.max() + 1] = new_row_for_header
#         final_df = df_cleaned.copy()
#         # Set the last row as the new column headers
#         new_header = final_df.iloc[-1]
#         # Remove the last row (which is now the header) from the DataFrame body
#         final_df = final_df.iloc[:-1]
#         # Assign the new headers to the DataFrame
#         final_df.columns = new_header

#         # Find data columns by looking for unit keywords in the new header
#         unit_keywords = ['kw', 'm3/min', 'rpm', 'bar', 'bar g']
#         last_unit_col_index = -1
#         # Iterate through the new column names (converted to string for safety)
#         for i, col_name in enumerate(final_df.columns.astype(str)):
#             # Check if any unit keyword is present in the column name (case-insensitive)
#             if any(keyword in col_name.lower() for keyword in unit_keywords):
#                 last_unit_col_index = i
        
#         if last_unit_col_index == -1:
#             raise ValueError("No unit keywords found in the generated header. Cannot identify data columns.")

#         # Data columns are considered to be after the last column containing a unit keyword
#         data_cols = final_df.columns[last_unit_col_index + 1:]
#         if data_cols.empty:
#             raise ValueError("No data columns identified after processing headers.")

#         # ==== Step 3: Extraction per model (for each data column) ====
#         output_rows = []
#         for col in data_cols:
#             row_data = {'Source File': file_name, 'Model': col, 'date_revised': date_objects}
#             row_data['Cooling Media'] = cooling_media
#             row_data['Hz'] = hz_value

#             # (A) Simple text-only fields extraction
#             for id_0 in simple_identifiers:
#                 match_col = None
#                 # Search for the identifier in the first 3 columns
#                 for i in range(3):
#                     if i < final_df.shape[1] and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_0, case=False, na=False).any():
#                         match_col = i
#                         break
                
#                 value = 'not found'
#                 if match_col is not None:
#                     # Create a boolean mask for rows where the identifier is found
#                     mask = final_df.iloc[:, match_col].astype(str).str.strip().str.contains(id_0, case=False, na=False)
#                     # Filter the DataFrame using the mask
#                     match = final_df[mask]
#                     # Get the value from the current 'col' (model column) if found
#                     if not match.empty and col in match.columns:
#                         value = match[col].values[0]
                
#                 display_key = id_0.title() # Convert identifier to title case for the column name
#                 row_data[display_key] = value

#             # (B) Fields needing text + unit match extraction
#             for id_0, id_2 in unit_identifiers:
#                 match_0 = match_2 = None
#                 # Search for the first identifier (id_0) in the first 3 columns
#                 for i in range(3):
#                     if i < final_df.shape[1] and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_0, case=False, na=False).any():
#                         match_0 = i
#                         break
#                 # Search for the second identifier (id_2) in the first 3 columns, excluding the column where id_0 was found
#                 for i in range(3):
#                     if i < final_df.shape[1] and i != match_0 and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_2, case=False, na=False).any():
#                         match_2 = i
#                         break

#                 # Create masks based on whether identifiers were found
#                 mask_0 = pd.Series([False] * len(final_df))
#                 if match_0 is not None:
#                     mask_0 = final_df.iloc[:, match_0].astype(str).str.strip().str.contains(id_0, case=False, na=False)
                
#                 mask_2 = pd.Series([True] * len(final_df)) # Default to True if no id_2 or no match for id_2
#                 if id_2 and match_2 is not None:
#                     mask_2 = final_df.iloc[:, match_2].astype(str).str.strip().str.contains(id_2, case=False, na=False)
                
#                 # Ensure masks have the same length as final_df to prevent alignment issues
#                 if len(mask_0) != len(final_df):
#                     mask_0 = pd.Series([False] * len(final_df))
#                 if len(mask_2) != len(final_df):
#                     mask_2 = pd.Series([False] * len(final_df))

#                 # Combine masks to find rows matching both criteria
#                 match = final_df[mask_0 & mask_2]
#                 value = 'not found'
#                 if not match.empty and col in match.columns:
#                     value = match[col].values[0]
                
#                 # Construct display key for the column name
#                 display_key = f"{id_0.title()}_{id_2.title()}" if id_2 else id_0.title()
#                 row_data[display_key] = value

#             output_rows.append(row_data)

#         if output_rows:
#             # If data was extracted, create a DataFrame
#             extracted_df = pd.DataFrame(output_rows)
#             compiled_results.append(extracted_df)
#         else:
#             # Log an error if no data was extracted for this file
#             error_logs.append(f"No data extracted for file '{file_name}' after initial processing.")

#     except Exception as e:
#         # Catch any exceptions during processing and log them
#         error_logs.append(f"Error processing file '{file_name}': {str(e)}")

#     # Return the concatenated results and any accumulated error messages
#     if compiled_results:
#         final_extracted_df = pd.concat(compiled_results, ignore_index=True)
#     else:
#         final_extracted_df = pd.DataFrame() # Return an empty DataFrame if nothing was extracted

#     return final_extracted_df, error_logs

# def concatenate_fields(row):
#     """
#     Concatenates various fields from a row to create a 'name' column,
#     replicating the logic from the notebook.
#     """
#     model_data = str(row.get('Performance Data_Performance Data', '')).strip()
#     cooling_media = str(row.get('Cooling Media', '')).strip()
#     hz = str(row.get('Hz', ''))
#     hz_suffix = f"({hz} Hz)" if hz else ''

#     bar_g = str(row.get('Working_pressure', '')).strip()
#     bar_suffix = (' bar') if bar_g else ''

#     # Logic to conditionally add cooling_media based on model_data ending
#     if not model_data.endswith(('A', 'W')):
#         model_data_with_cooling = ' '.join([model_data, cooling_media]) if cooling_media else model_data
#     else:
#         model_data_with_cooling = model_data

#     # Join all parts, filtering out empty strings
#     return ' '.join(part for part in [model_data_with_cooling, hz_suffix, bar_g, bar_suffix] if part)


# def app():
#     st.set_page_config(layout="wide", page_title="Excel Data Extractor")

#     st.title("📄 Excel Data Extractor for Fixed Speed Models")
#     st.markdown("""
#         Upload Excel datasheets, and this app will extract and transform specific performance data
#         into a structured CSV file for download.
#     """)

#     # Step 1: Ask user for file type (Fix speed or VSD)
#     file_type_choice = st.radio(
#         "Select the type of compressor data you are uploading:",
#         ("Fix speed", "VSD")
#     )

#     if file_type_choice == "Fix speed":
#         # Step 2: Ask user for Manufacturer name
#         manufacturer_name = st.text_input(
#             "Enter the Manufacturer Name for these files:",
#             value="CompAir", # Default value as per original code
#             help="This name will be added to the 'Manufacturer' column in the output CSV."
#         )

#         # Step 3: Allow multiple file uploads
#         uploaded_files = st.file_uploader(
#             "Choose Excel files (.xls, .xlsx)",
#             type=["xls", "xlsx"],
#             accept_multiple_files=True,
#             help="Select one or more Excel files from your local machine to begin processing."
#         )

#         if uploaded_files:
#             all_extracted_dfs = []
#             all_processing_errors = []
            
#             progress_bar = st.progress(0)
#             status_text = st.empty()

#             for i, uploaded_file in enumerate(uploaded_files):
#                 status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: '{uploaded_file.name}'...")
#                 progress_bar.progress((i + 1) / len(uploaded_files))

#                 try:
#                     # Read the uploaded Excel file into a pandas DataFrame.
#                     # The original notebook code used `header=None`, so we maintain that.
#                     df_uploaded_raw = pd.read_excel(uploaded_file, header=None)
                    
#                     # Process the uploaded DataFrame using the notebook's core logic
#                     extracted_df, processing_errors_for_file = process_excel_file(df_uploaded_raw, uploaded_file.name)
                    
#                     if not extracted_df.empty:
#                         all_extracted_dfs.append(extracted_df)
                    
#                     if processing_errors_for_file:
#                         all_processing_errors.extend(processing_errors_for_file)

#                 except Exception as e:
#                     all_processing_errors.append(f"Error reading file '{uploaded_file.name}': {e}")
            
#             progress_bar.empty()
#             status_text.empty()

#             if all_processing_errors:
#                 st.warning("Some errors occurred during processing. Please download the error log for details.")
#                 # Create a DataFrame for errors
#                 errors_df = pd.DataFrame({"Timestamp": [datetime.now()] * len(all_processing_errors),
#                                           "Error Message": all_processing_errors})
                
#                 # Convert error DataFrame to CSV for download
#                 error_csv_buffer = io.StringIO()
#                 errors_df.to_csv(error_csv_buffer, index=False)
#                 error_csv_content_bytes = error_csv_buffer.getvalue().encode('utf-8')

#                 st.download_button(
#                     label="⬇️ Download Error Log (CSV)",
#                     data=error_csv_content_bytes,
#                     file_name="processing_error_log.csv",
#                     mime="text/csv",
#                     help="Click to download a CSV file containing all processing errors."
#                 )
            
#             if all_extracted_dfs:
#                 st.success("Data extraction complete! Applying final transformations...")
                
#                 # Concatenate all extracted DataFrames
#                 final_extracted_df = pd.concat(all_extracted_dfs, ignore_index=True)

#                 # st.subheader("Extracted Data Preview (Initial):")
#                 # st.dataframe(final_extracted_df.head())

#                 # Apply the additional processing steps from the notebook (Working_pressure, concatenated_field)
#                 # Ensure 'Model' column is string type before rsplit
#                 final_extracted_df['Working_pressure'] = final_extracted_df['Model'].astype(str).str.rsplit('_', n=1).str[-1]
#                 final_extracted_df['concatenated_field'] = final_extracted_df.apply(concatenate_fields, axis=1)

#                 # Rename columns as per the notebook
#                 all_files_combined = final_extracted_df.copy()
#                 all_files_combined = all_files_combined.rename(columns={
#                     '5 Bar_M3/Min': 'Flow Level 1 (5bar)',
#                     '6 Bar_M3/Min': 'Flow Level 2 (6bar)',
#                     '7 Bar_M3/Min': 'Flow Level 3 (7bar)',
#                     '7.5 Bar_M3/Min': 'Flow Level 4 (7.5bar)',
#                     '8 Bar_M3/Min': 'Flow Level 5 (8bar)',
#                     '9 Bar_M3/Min': 'Flow Level 6 (9bar)',
#                     '10 Bar_M3/Min': 'Flow Level 7 (10bar)',
#                     '11 Bar_M3/Min': 'Flow Level 8 (11bar)',
#                     '5 Bar_Kw': 'Kw Level 1 (5 bar)',
#                     '6 Bar_Kw': 'Kw Level 2 (6 bar)',
#                     '7 Bar_Kw': 'Kw Level 3 (7 bar)',
#                     '7.5 Bar_Kw': 'Kw Level 4 (7.5 bar)',
#                     '8 Bar_Kw': 'Kw Level 5 (8 bar)',
#                     '9 Bar_Kw': 'Kw Level 6 (9 bar)',
#                     '10 Bar_Kw': 'Kw Level 7 (10 bar)',
#                     '11 Bar_Kw': 'Kw Level 8 (11 bar)',
#                     'No Load': 'Unload Kw',
#                     '_Kw' : 'max kw', # This column name might vary based on actual data
#                     'Working_pressure':'Maximal Setpoint Pressure',
#                     'concatenated_field': 'name',
#                     'Performance Data_Performance Data': 'model_code',
#                 })

#                 # Drop specified columns from the notebook.
#                 # Use a list comprehension to only drop columns that actually exist in the DataFrame.
#                 cols_to_drop_final = ['Performance Data_Bar G', 'Maximum Working Pressure_Bar G']
#                 all_files_combined.drop(columns=[col for col in cols_to_drop_final if col in all_files_combined.columns], inplace=True)

#                 # Add RATED FLOW and MANUFACTURER columns
#                 flow_cols = [c for c in all_files_combined.columns if "Flow Level" in c]
#                 # Convert flow columns to numeric, coercing errors to NaN
#                 all_files_combined[flow_cols] = all_files_combined[flow_cols].apply(pd.to_numeric, errors="coerce")
#                 # Calculate min flow across flow level columns
#                 all_files_combined["flow_at_maxworking_press"] = all_files_combined[flow_cols].min(axis=1, skipna=True)
#                 all_files_combined['kw_at_maxworking_press'] = all_files_combined['max kw']
#                 all_files_combined['Manufacturer'] = manufacturer_name # Use the user-inputted manufacturer name

#                 st.success("All transformations applied successfully!")
#                 st.subheader("Final Processed Data Preview (Ready for Download):")
#                 st.dataframe(all_files_combined.head())

#                 # Convert the final DataFrame to CSV format in memory for download
#                 csv_buffer = io.StringIO()
#                 all_files_combined.to_csv(csv_buffer, index=False)
#                 csv_content_bytes = csv_buffer.getvalue().encode('utf-8')

#                 st.download_button(
#                     label="⬇️ Download extracted_cagi_compilation.csv",
#                     data=csv_content_bytes,
#                     file_name="extracted_cagi_compilation.csv",
#                     mime="text/csv",
#                     help="Click to download the final processed data as a CSV file."
#                 )
#             else:
#                 st.info("No data could be extracted from any of the uploaded files based on the defined logic. "
#                         "Please ensure the Excel file structures match the expected format for extraction.")

#         else:
#             st.info("Please upload one or more Excel files to start the data extraction process.")

#     else: # If VSD is chosen
#         st.info("Currently, this application only supports 'Fix speed' compressor data. Please select 'Fix speed' to proceed.")

# if __name__ == "__main__":
#     app()


import streamlit as st
import pandas as pd
import re
from datetime import datetime
import io

def process_excel_file(df_uploaded, file_name):
    """
    Processes a single pandas DataFrame (from an uploaded Excel file or a single sheet)
    according to the logic provided in the notebook code.
    Returns the processed DataFrame and any error messages.
    """
    compiled_results = []
    error_logs = []

    # ==== Step 1: Prepare Identifiers ====
    # (A) Only text match needed
    simple_identifiers = [
        'no load',
        'Male rotor speed',
        'Nominal motor rating',
        'Drive motor speed',
        'Fan motor rating'
    ]
    simple_identifiers = [id_.strip().lower() for id_ in simple_identifiers]

    # (B) Text + Unit match needed
    unit_identifiers = [
        ('5 bar', 'm3/min'), ('5 bar', 'KW'), ('6 bar', 'm3/min'), ('6 bar', 'KW'),
        ('7 bar', 'm3/min'), ('7 bar', 'KW'), ('7.5 bar', 'm3/min'), ('7.5 bar', 'KW'),
        ('8 bar', 'm3/min'), ('8 bar', 'KW'), ('9 bar', 'm3/min'), ('9 bar', 'KW'),
        ('10 bar', 'm3/min'), ('10 bar', 'KW'), ('11 bar', 'm3/min'), ('11 bar', 'KW'),
        ('', 'kW'), # Note: This is for a standalone 'kW' unit
        ('Performance Data', 'Performance Data'), # Special case for header
        ('Performance Data', 'bar g'), # Special case for header
        ('Maximum working pressure', 'bar g') # Special case for header
    ]
    unit_identifiers = [(i[0].strip().lower(), i[1].strip().lower()) for i in unit_identifiers]

    try:
        df = df_uploaded.copy() # Work on a copy of the uploaded DataFrame
        
        # Initial check for empty or too few rows
        if df.empty or len(df) < 5:
            raise ValueError("Empty or too few rows in the uploaded file/sheet. Minimum 5 rows required.")

        # === 1. Cooling Media detection FIRST ===
        # Convert top 15 rows to string and lowercase for robust searching
        top_rows_raw = df.head(15).astype(str).apply(lambda x: x.str.lower())

        cooling_media = 'Unknown'
        if top_rows_raw.apply(lambda x: x.str.contains('water', na=False).any()).any():
            cooling_media = 'W'
        elif top_rows_raw.apply(lambda x: x.str.contains('air', na=False).any()).any():
            cooling_media = 'A'

        # Detect Hz value (50 or 60 Hz)
        hz_value = None
        # Iterate through all cell values in the top rows
        for cell_value in top_rows_raw.values.flatten():
            match = re.search(r'\b(50|60)\s*hz\b', str(cell_value), re.IGNORECASE)
            if match:
                hz_value = int(match.group(1)) # Extract 50 or 60
                break

        # Drop columns where the value in the 5th row (index 4) is NaN
        # This assumes the 5th row (index 4) is crucial for identifying valid data columns
        cols_to_drop = [col for col in df.columns[1:] if pd.isna(df.loc[4, col])]
        df_cleaned = df.drop(columns=cols_to_drop)

        # Detect date row using multiple patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}/\d{2}/\d{2}', r'\b\d{1,2}[ ]?[A-Za-z]{3,9}[ ]?\d{4}\b',
            r'\b[A-Za-z]{3,9}[ ]?\d{1,2},?[ ]?\d{4}\b'
        ]
        combined_pattern = '|'.join(date_patterns)
        # Check each row if any cell contains a date pattern
        date_row_mask = df_cleaned.apply(lambda row: row.astype(str).str.contains(combined_pattern, regex=True, case=False, na=False).any(), axis=1)

        if not date_row_mask.any():
            raise ValueError(f"No date row found in '{file_name}'. Please ensure a date is present in a row.")
        date_row_index = df_cleaned[date_row_mask].index[0] # Get the index of the first row containing a date

        original_row = df_cleaned.loc[date_row_index].copy()
        # Create a 'cleaned' version of the date row, masking out the date string itself
        cleaned_row = original_row.mask(original_row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}'))
        # Concatenate this 'cleaned' row back to the DataFrame.
        # This part of the original notebook code is a bit unusual but maintained for consistency.
        df_cleaned = pd.concat([df_cleaned, pd.DataFrame([cleaned_row], columns=df_cleaned.columns)], ignore_index=True)

        # Extract actual date objects from the original date row
        date_strings = original_row[original_row.astype(str).str.contains(r'\d{4}-\d{2}-\d{2}')].astype(str)
        date_objects = pd.to_datetime(date_strings, errors='coerce').dropna().tolist()

        # Forward fill specific rows for data propagation
        # Ensure indices exist before attempting to fill
        rows_to_fill_indices = [idx for idx in [3, 5, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, df_cleaned.index[-1]] if idx in df_cleaned.index]
        if rows_to_fill_indices:
            df_cleaned.loc[rows_to_fill_indices] = df_cleaned.loc[rows_to_fill_indices].ffill(axis=1)
        # Forward fill the first column (index 0)
        df_cleaned.iloc[:, 0] = df_cleaned.iloc[:, 0].ffill()

        # Build new header by concatenating values from row 3 and row 4
        # This assumes row 3 and 4 contain parts of the desired column names
        new_row_for_header = (df_cleaned.iloc[3].astype(str) + '_' + df_cleaned.iloc[4].astype(str))
        # Add this new header row to the end of the DataFrame temporarily
        df_cleaned.loc[df_cleaned.index.max() + 1] = new_row_for_header
        final_df = df_cleaned.copy()
        # Set the last row as the new column headers
        new_header = final_df.iloc[-1]
        # Remove the last row (which is now the header) from the DataFrame body
        final_df = final_df.iloc[:-1]
        # Assign the new headers to the DataFrame
        final_df.columns = new_header

        # Find data columns by looking for unit keywords in the new header
        unit_keywords = ['kw', 'm3/min', 'rpm', 'bar', 'bar g']
        last_unit_col_index = -1
        # Iterate through the new column names (converted to string for safety)
        for i, col_name in enumerate(final_df.columns.astype(str)):
            # Check if any unit keyword is present in the column name (case-insensitive)
            if any(keyword in col_name.lower() for keyword in unit_keywords):
                last_unit_col_index = i
        
        if last_unit_col_index == -1:
            raise ValueError(f"No unit keywords found in the generated header for '{file_name}'. Cannot identify data columns.")

        # Data columns are considered to be after the last column containing a unit keyword
        data_cols = final_df.columns[last_unit_col_index + 1:]
        if data_cols.empty:
            raise ValueError(f"No data columns identified after processing headers for '{file_name}'.")

        # ==== Step 3: Extraction per model (for each data column) ====
        output_rows = []
        for col in data_cols:
            row_data = {'Source File': file_name, 'Model': col, 'date_revised': date_objects}
            row_data['Cooling Media'] = cooling_media
            row_data['Hz'] = hz_value

            # (A) Simple text-only fields extraction
            for id_0 in simple_identifiers:
                match_col = None
                # Search for the identifier in the first 3 columns
                for i in range(3):
                    if i < final_df.shape[1] and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_0, case=False, na=False).any():
                        match_col = i
                        break
                
                value = 'not found'
                if match_col is not None:
                    # Create a boolean mask for rows where the identifier is found
                    mask = final_df.iloc[:, match_col].astype(str).str.strip().str.contains(id_0, case=False, na=False)
                    # Filter the DataFrame using the mask
                    match = final_df[mask]
                    # Get the value from the current 'col' (model column) if found
                    if not match.empty and col in match.columns:
                        value = match[col].values[0]
                
                display_key = id_0.title() # Convert identifier to title case for the column name
                row_data[display_key] = value

            # (B) Fields needing text + unit match extraction
            for id_0, id_2 in unit_identifiers:
                match_0 = match_2 = None
                # Search for the first identifier (id_0) in the first 3 columns
                for i in range(3):
                    if i < final_df.shape[1] and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_0, case=False, na=False).any():
                        match_0 = i
                        break
                # Search for the second identifier (id_2) in the first 3 columns, excluding the column where id_0 was found
                for i in range(3):
                    if i < final_df.shape[1] and i != match_0 and final_df.iloc[:, i].astype(str).str.strip().str.contains(id_2, case=False, na=False).any():
                        match_2 = i
                        break

                # Create masks based on whether identifiers were found
                mask_0 = pd.Series([False] * len(final_df))
                if match_0 is not None:
                    mask_0 = final_df.iloc[:, match_0].astype(str).str.strip().str.contains(id_0, case=False, na=False)
                
                mask_2 = pd.Series([True] * len(final_df)) # Default to True if no id_2 or no match for id_2
                if id_2 and match_2 is not None:
                    mask_2 = final_df.iloc[:, match_2].astype(str).str.strip().str.contains(id_2, case=False, na=False)
                
                # Ensure masks have the same length as final_df to prevent alignment issues
                if len(mask_0) != len(final_df):
                    mask_0 = pd.Series([False] * len(final_df))
                if len(mask_2) != len(final_df):
                    mask_2 = pd.Series([False] * len(final_df))

                # Combine masks to find rows matching both criteria
                match = final_df[mask_0 & mask_2]
                value = 'not found'
                if not match.empty and col in match.columns:
                    value = match[col].values[0]
                
                # Construct display key for the column name
                display_key = f"{id_0.title()}_{id_2.title()}" if id_2 else id_0.title()
                row_data[display_key] = value

            output_rows.append(row_data)

        if output_rows:
            # If data was extracted, create a DataFrame
            extracted_df = pd.DataFrame(output_rows)
            compiled_results.append(extracted_df)
        else:
            # Log an error if no data was extracted for this file
            error_logs.append(f"No data extracted for file '{file_name}' after initial processing.")

    except Exception as e:
        # Catch any exceptions during processing and log them
        error_logs.append(f"Error processing file '{file_name}': {str(e)}")

    # Return the concatenated results and any accumulated error messages
    if compiled_results:
        final_extracted_df = pd.concat(compiled_results, ignore_index=True)
    else:
        final_extracted_df = pd.DataFrame() # Return an empty DataFrame if nothing was extracted

    return final_extracted_df, error_logs

def concatenate_fields(row):
    """
    Concatenates various fields from a row to create a 'name' column,
    replicating the logic from the notebook.
    """
    model_data = str(row.get('Performance Data_Performance Data', '')).strip()
    cooling_media = str(row.get('Cooling Media', '')).strip()
    hz = str(row.get('Hz', ''))
    hz_suffix = f"({hz} Hz)" if hz else ''

    bar_g = str(row.get('Working_pressure', '')).strip()
    bar_suffix = (' bar') if bar_g else ''

    # Logic to conditionally add cooling_media based on model_data ending
    if not model_data.endswith(('A', 'W')):
        model_data_with_cooling = ' '.join([model_data, cooling_media]) if cooling_media else model_data
    else:
        model_data_with_cooling = model_data

    # Join all parts, filtering out empty strings
    return ' '.join(part for part in [model_data_with_cooling, hz_suffix, bar_g, bar_suffix] if part)


def app():
    st.set_page_config(layout="wide", page_title="Excel Data Extractor")

    st.title("📄 Excel Data compiler for Fixed Speed Models")
    st.markdown("""
        Upload Excel datasheets which contain fix speed compressors datasheet, and this app will extract and transform specific performance data
        into a structured CSV file for download.
    """)

    # Step 1: Ask user for file type (Fix speed or VSD)
    file_type_choice = st.radio(
        "Select the type of compressor data you are uploading:",
        ("Fix speed", "VSD")
    )

    if file_type_choice == "Fix speed":
        # Step 2: Ask user for Manufacturer name
        manufacturer_name = st.text_input(
            "Enter the Manufacturer Name for these files:",
            value="CompAir", # Default value as per original code
            help="This name will be added to the 'Manufacturer' column in the output CSV."
        )

        # Step 3: Allow multiple file uploads
        uploaded_files = st.file_uploader(
            "Choose Excel files (.xls, .xlsx)",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
            help="Select one or more Excel files from your local machine to begin processing."
        )

        if uploaded_files:
            all_extracted_dfs = []
            all_processing_errors = []
            judgement_word = 'speed 2' # Define the keyword for VSD sheets
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            processed_files_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                processed_files_count += 1
                status_text.text(f"Processing file {processed_files_count}/{total_files}: '{uploaded_file.name}'...")
                progress_bar.progress(processed_files_count / total_files)

                try:
                    # Read all sheets from the uploaded Excel file
                    xls = pd.read_excel(uploaded_file, sheet_name=None, header=None)
                    base_file_name = uploaded_file.name.split('.')[0] # Get name without extension

                    for sheet_name, df_sheet in xls.items():
                        # Construct a unique name for the sheet for logging/identification
                        sheet_identifier = f"{base_file_name}@{sheet_name}.xlsx"

                        # Check for 'judgement_word' in the sheet
                        # Convert entire DataFrame to string to search all cells
                        contains_judgement_word = df_sheet.astype(str).apply(
                            lambda x: x.str.contains(judgement_word, case=False, na=False)
                        ).any().any()

                        if not contains_judgement_word: # This is a 'Fix speed' sheet
                            extracted_df, processing_errors_for_sheet = process_excel_file(df_sheet, sheet_identifier)
                            
                            if not extracted_df.empty:
                                all_extracted_dfs.append(extracted_df)
                            
                            if processing_errors_for_sheet:
                                all_processing_errors.extend(processing_errors_for_sheet)
                        else: # This is a 'VSD' sheet, log it as skipped
                            all_processing_errors.append(
                                f"Skipped sheet '{sheet_identifier}' as it contains '{judgement_word}' (VSD data)."
                            )

                except Exception as e:
                    all_processing_errors.append(f"Error processing '{uploaded_file.name}': {e}")
            
            progress_bar.empty()
            status_text.empty()

            if all_processing_errors:
                st.warning("Some errors occurred during processing. Please download the error log for details.")
                # Create a DataFrame for errors
                errors_df = pd.DataFrame({
                    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(all_processing_errors),
                    "Error Message": all_processing_errors
                })
                
                # Convert error DataFrame to CSV for download
                error_csv_buffer = io.StringIO()
                errors_df.to_csv(error_csv_buffer, index=False)
                error_csv_content_bytes = error_csv_buffer.getvalue().encode('utf-8')

                st.download_button(
                    label="⬇️ Download Error Log (CSV)",
                    data=error_csv_content_bytes,
                    file_name="processing_error_log.csv",
                    mime="text/csv",
                    help="Click to download a CSV file containing all processing errors."
                )
            
            if all_extracted_dfs:
                st.success("Data extraction complete! Applying final transformations...")
                
                # Concatenate all extracted DataFrames
                final_extracted_df = pd.concat(all_extracted_dfs, ignore_index=True)

                # st.subheader("Extracted Data Preview (Initial):")
                # st.dataframe(final_extracted_df.head())

                # Apply the additional processing steps from the notebook (Working_pressure, concatenated_field)
                # Ensure 'Model' column is string type before rsplit
                final_extracted_df['Working_pressure'] = final_extracted_df['Model'].astype(str).str.rsplit('_', n=1).str[-1]
                final_extracted_df['concatenated_field'] = final_extracted_df.apply(concatenate_fields, axis=1)

                # Rename columns as per the notebook
                all_files_combined = final_extracted_df.copy()
                all_files_combined = all_files_combined.rename(columns={
                    '5 Bar_M3/Min': 'Flow Level 1 (5bar)',
                    '6 Bar_M3/Min': 'Flow Level 2 (6bar)',
                    '7 Bar_M3/Min': 'Flow Level 3 (7bar)',
                    '7.5 Bar_M3/Min': 'Flow Level 4 (7.5bar)',
                    '8 Bar_M3/Min': 'Flow Level 5 (8bar)',
                    '9 Bar_M3/Min': 'Flow Level 6 (9bar)',
                    '10 Bar_M3/Min': 'Flow Level 7 (10bar)',
                    '11 Bar_M3/Min': 'Flow Level 8 (11bar)',
                    '5 Bar_Kw': 'Kw Level 1 (5 bar)',
                    '6 Bar_Kw': 'Kw Level 2 (6 bar)',
                    '7 Bar_Kw': 'Kw Level 3 (7 bar)',
                    '7.5 Bar_Kw': 'Kw Level 4 (7.5 bar)',
                    '8 Bar_Kw': 'Kw Level 5 (8 bar)',
                    '9 Bar_Kw': 'Kw Level 6 (9 bar)',
                    '10 Bar_Kw': 'Kw Level 7 (10 bar)',
                    '11 Bar_Kw': 'Kw Level 8 (11 bar)',
                    'No Load': 'Unload Kw',
                    '_Kw' : 'max kw', # This column name might vary based on actual data
                    'Working_pressure':'Maximal Setpoint Pressure',
                    'concatenated_field': 'name',
                    'Performance Data_Performance Data': 'model_code',
                })

                # Drop specified columns from the notebook.
                # Use a list comprehension to only drop columns that actually exist in the DataFrame.
                cols_to_drop_final = ['Performance Data_Bar G', 'Maximum Working Pressure_Bar G']
                all_files_combined.drop(columns=[col for col in cols_to_drop_final if col in all_files_combined.columns], inplace=True)

                # Add RATED FLOW and MANUFACTURER columns
                flow_cols = [c for c in all_files_combined.columns if "Flow Level" in c]
                # Convert flow columns to numeric, coercing errors to NaN
                all_files_combined[flow_cols] = all_files_combined[flow_cols].apply(pd.to_numeric, errors="coerce")
                # Calculate min flow across flow level columns
                all_files_combined["flow_at_maxworking_press"] = all_files_combined[flow_cols].min(axis=1, skipna=True)
                all_files_combined['kw_at_maxworking_press'] = all_files_combined['max kw']
                all_files_combined['Manufacturer'] = manufacturer_name # Use the user-inputted manufacturer name

                st.success("All transformations applied successfully!")
                st.subheader("Final Processed Data Preview (Ready for Download):")
                st.dataframe(all_files_combined.head())

                # Convert the final DataFrame to CSV format in memory for download
                csv_buffer = io.StringIO()
                all_files_combined.to_csv(csv_buffer, index=False)
                csv_content_bytes = csv_buffer.getvalue().encode('utf-8')

                st.download_button(
                    label="⬇️ Download extracted_cagi_compilation.csv",
                    data=csv_content_bytes,
                    file_name="extracted_cagi_compilation.csv",
                    mime="text/csv",
                    help="Click to download the final processed data as a CSV file."
                )
            else:
                st.info("No data could be extracted from any of the uploaded files based on the defined logic. "
                        "Please ensure the Excel file structures match the expected format for extraction, and that they are 'Fix speed' type.")

        else:
            st.info("Please upload one or more Excel files to start the data extraction process.")

    else: # If VSD is chosen
        st.info("Currently, this application only supports 'Fix speed' compressor data. Please select 'Fix speed' to proceed.")

if __name__ == "__main__":
    app()

