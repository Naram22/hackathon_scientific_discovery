import requests
import zipfile
import io
import os
from datetime import datetime


# --- Configuration for 10 Years of Data ---
# Start year for the data download (inclusive)
START_YEAR = 2024
# End year for the data download (inclusive, going up to the last quarter of this year)
END_YEAR = datetime.now().year


# The base URL pattern for FAERS ASCII quarterly data files
BASE_URL = "https://fis.fda.gov/content/Exports/faers_ascii_{year}q{quarter}.zip"


# The essential files (tables) needed for drug-drug interaction analysis.
ESSENTIAL_FILES = [
   "DRUG",  # Drug information (for identifying combinations)
   "REAC",  # Adverse Reactions (the outcomes)
   "DEMO",  # Demographics (contains the main CASEID to link all tables)
   "INDI",  # Indications (useful context for why the drug was prescribed)
   "OUTC",  # Outcomes (e.g., death, hospitalization)
]


def download_and_extract_quarter(year, quarter):
   """
   Downloads and extracts the essential FAERS files for a single year and quarter.
   """
   download_url = BASE_URL.format(year=year, quarter=quarter)
  
   print(f"\n--- Processing {year} Q{quarter} ---")
   print(f"Source URL: {download_url}")


   try:
       # 1. Fetch the data from the FDA source
       response = requests.get(download_url, stream=True, timeout=600) # Increased timeout for large files
       response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)


       # Use io.BytesIO to handle the ZIP content in memory
       zip_file = zipfile.ZipFile(io.BytesIO(response.content))
      
       # Output directory setup for this specific quarter
       output_dir = f"faers_data/{year}_Q{quarter}"
       os.makedirs(output_dir, exist_ok=True)
       print(f"Data will be extracted to: {output_dir}")


       # 2. Extract only the essential ASCII files
       extracted_count = 0
       file_list = zip_file.namelist()
      
       for file_info in file_list:
           # Check if the file is one of our essential tables and is an ASCII/TXT file.
           base_filename = os.path.basename(file_info).upper()
          
           # The files inside the ZIP are often prefixed with 'ASCII/' or similar, and have a 2-digit year/quarter suffix.
           if base_filename.endswith('.TXT') and base_filename.startswith(tuple(ESSENTIAL_FILES)):
              
               # Find the core table name (e.g., 'DRUG' from 'DRUG24Q4.TXT')
               core_name = next((name for name in ESSENTIAL_FILES if base_filename.startswith(name)), None)
              
               if core_name:
                   # Rename the extracted file to a clean, consistent name (e.g., DRUG.txt)
                   new_filename = f"{core_name}.txt"
                  
                   # Extract the file content
                   with zip_file.open(file_info) as source, open(os.path.join(output_dir, new_filename), 'wb') as target:
                       target.write(source.read())
                  
                   print(f"   -> Extracted: {new_filename}")
                   extracted_count += 1
      
       print(f"✅ Success! {extracted_count} essential files extracted for {year} Q{quarter}.")


   except requests.exceptions.HTTPError as e:
       # 404 errors are expected if a quarter hasn't been released yet (e.g., 2025 Q4)
       if response.status_code == 404:
           print(f"⚠️ Warning: File not found (404). Data for {year} Q{quarter} is likely not yet publicly released.")
       else:
           print(f"❌ ERROR: HTTP Request Failed for {year} Q{quarter}.")
           print(f"Detail: {e}")
   except requests.exceptions.RequestException as e:
       print(f"❌ ERROR: Network or Request Error for {year} Q{quarter}. Detail: {e}")
   except Exception as e:
       print(f"❌ An unexpected error occurred during extraction for {year} Q{quarter}: {e}")


def main():
   """
   Generates the list of quarters from START_YEAR Q1 to END_YEAR Q4 and runs the downloader.
   """
   all_quarters = []
   for year in range(START_YEAR, END_YEAR + 1):
       for quarter in range(1, 5): # Quarters 1 through 4
           all_quarters.append((year, quarter))
  
   print(f"Starting batch download of {len(all_quarters)} FAERS quarters, from {START_YEAR} Q1 to {END_YEAR} Q4.")
  
   for year, quarter in all_quarters:
       download_and_extract_quarter(year, quarter)


   print("\n\n=============== BATCH DOWNLOAD PROCESS COMPLETE ===============")
   print(f"All files are organized in the 'faers_data' directory.")
   print("Remember to process the contents of each quarter separately for the deduplication and standardization steps.")


if __name__ == "__main__":
   # Ensure you have the 'requests' library: pip install requests
   main()