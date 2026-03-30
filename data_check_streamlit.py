"""
Calypso vs instO Data Reconciliation Tool - Streamlit Web Version
Run with: streamlit run data_check_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io
import base64
import logging
from typing import Dict, Tuple

# Must be first Streamlit command
st.set_page_config(
    page_title="Data Reconciliation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Configuration (same as original)
# ============================================================================

BOOK_TO_ACCOUNT = {
    'SWHYI_PROP_FICC_IHFIASIA': 'IHFIASIA',
    'SWHYI_GM_MAD_IHMABOND_SB': 'IHMABOND',
    'SWHYI_GM_EQD_IHEQSWAP_SB': 'IHEQSWAP',
    'SWHYI_GM_EQD_IHEQTRX_SB': 'IHEQTRX',
    'SWHYI_GM_EQD_IHFUSWAP_SB': 'IHFUSWAP',
    'SWHYI_GM_EQD_IHEQPROP': 'IHEQPROP',
    'SWHYI_PROP_FICC_IHFIPROP': 'IHFIPROP',
    'SWHYI_GM_EQD_IHEQBSHR_NB': 'IHEQBSHR',
    'SWHYI_GM_EQD_IHEQFUT': 'IHEQFUT',
    'SWHYI_PROP_FICC_IHFITRAD': 'IHFITRAD',
    'SWHYI_GM_EQD_IHEQCASH': 'IHEQCASH',
    'SWHYI_GM_MAD_IHMAFIFU_SB': 'IHMAFIFU',
    'SWHYI_GM_EQD_IHEQQFII_NB': 'IHEQQFII',
    'SWHYI_MGT_Treasury_IHCASH': 'IHCASH',
    'SWHYI_GM_EQD_IHEQSGPB_SB': 'IHEQSGPB'
}

ACCOUNT_TO_BOOK = {v: k for k, v in BOOK_TO_ACCOUNT.items()}

# ============================================================================
# Data Processing Functions (same logic as GUI version)
# ============================================================================

@st.cache_data
def load_calypso_data(file_bytes):
    """Load Calypso data from uploaded file"""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin-1')
        return df
    except Exception as e:
        st.error(f"Failed to load Calypso data: {e}")
        return None

@st.cache_data
def load_insto_data(file_bytes):
    """Load instO data from uploaded file"""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), header=6)
        return df
    except Exception as e:
        st.error(f"Failed to load instO data: {e}")
        return None

def filter_insto_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Security'] = df['Security'].astype(str)
    df['Account Code'] = df['Account Code'].astype(str)
    
    cond_total = df['Security'] == 'Total'
    cond_cash = df['Security'].str.startswith('{cash}', na=False)
    cond_accrual = df['Security'].str.startswith('{accrual}', na=False)
    cond_empty = df['Security'] == '[]'
    cond_test = df['Account Code'].isin(['IHTESTEQ', 'IHTESTFI'])
    
    cond_final = cond_total | cond_cash | cond_accrual | cond_empty | cond_test
    return df[~cond_final].copy()

def filter_calypso_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Product Type'] = df['Product Type'].astype(str)
    
    cond_bond = df['Product Type'] == 'Bond'
    cond_equity = df['Product Type'] == 'Equity'
    cond_future = df['Product Type'].str.contains('Future', na=False, regex=False)
    cond_final = cond_bond | cond_equity | cond_future
    
    return df[cond_final].copy()

def add_book_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Book_from_Account'] = df['Account Code'].map(ACCOUNT_TO_BOOK)
    return df

def add_simplified_isin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['PRODUCT_CODE.ISIN'] = df['PRODUCT_CODE.ISIN'].replace({
        'XUCM6 Curncy': 'SGXDB1216058'
    })
    df['simplified_ISIN'] = df['PRODUCT_CODE.ISIN'].str.split().str[0]
    return df

def split_insto_by_isin(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    isin_not_null = df['ISIN'].notna()
    return df[isin_not_null].copy(), df[~isin_not_null].copy()

def merge_with_isin(df_inst: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge(
        df_inst,
        df_cal[['PRODUCT_CODE.ISIN', 'Book', 'SWHYI_QUANTITY', 'simplified_ISIN']],
        left_on=['ISIN', 'Book_from_Account'],
        right_on=['PRODUCT_CODE.ISIN', 'Book'],
        how='left'
    )
    result.rename(columns={'SWHYI_QUANTITY': 'Calypso Quantity', 'Book': 'Calypso Book'}, inplace=True)
    return result

def merge_without_isin(df_inst: pd.DataFrame, df_cal: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge(
        df_inst,
        df_cal[['simplified_ISIN', 'Book', 'SWHYI_QUANTITY', 'PRODUCT_CODE.ISIN']],
        left_on=['Ticker', 'Book_from_Account'],
        right_on=['simplified_ISIN', 'Book'],
        how='left'
    )
    result.rename(columns={'SWHYI_QUANTITY': 'Calypso Quantity', 'Book': 'Calypso Book'}, inplace=True)
    return result

def combine_merged_results(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    cols1, cols2 = set(df1.columns), set(df2.columns)
    for col in cols1 - cols2:
        df2[col] = None
    for col in cols2 - cols1:
        df1[col] = None
    return pd.concat([df1, df2], ignore_index=True)

def analyze_quantity_matches(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['Position'] = pd.to_numeric(result['Position'], errors='coerce').fillna(0)
    result['Calypso Quantity'] = pd.to_numeric(result['Calypso Quantity'], errors='coerce').fillna(0)
    result['Security'] = result['Security'].astype(str)
    
    future_cases = ['TYM6 CBT', 'USM6 CBT', 'TUM6 CBT']
    
    def calc_var(row):
        pos = row['Position']
        cal = row['Calypso Quantity']
        sec = row['Security']
        acc = row['Account Code']
        
        if pos < 0 and sec.endswith('RP'):
            return 'REPO'
        
        if acc == 'IHMABOND' and abs(cal * 10 - pos) < 0.001:
            cal = cal * 10
        
        if sec in future_cases and abs(cal - pos * 1000) < 0.001:
            cal = cal / 1000
            
        if sec=='XUCM6 SGX'and abs(cal * 100000 - pos) < 0.001:
            cal = cal * 100000
        
        return pos - cal
    
    result['var'] = result.apply(calc_var, axis=1)
    return result

def generate_excel_download(df: pd.DataFrame) -> str:
    """Generate Excel file for download"""
    cols = ['Security', 'Account Code', 'ISIN', 'Ticker', 'Position', 'Calypso Quantity', 'var']
    existing = [c for c in cols if c in df.columns]
    result = df[existing].copy()
    
    def check_status(x):
        if x == 'REPO':
            return 'Match'
        if isinstance(x, (int, float)) and abs(x) < 0.001:
            return 'Match'
        return 'Mismatch'
    
    result['Status'] = result['var'].apply(check_status)
    
    # Create Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result.to_excel(writer, index=False, sheet_name='Reconciliation')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    return b64, result

# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.title("📊 Calypso vs instO Data Reconciliation Tool")
    st.markdown("Upload your Calypso CSV and instO Excel files to generate reconciliation report.")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 File Upload")
        
        calypso_file = st.file_uploader(
            "Calypso File (CSV)", 
            type=['csv'],
            help="Upload Calypso CSV export file"
        )
        
        insto_file = st.file_uploader(
            "instO File (Excel)", 
            type=['xlsx', 'xls'],
            help="Upload instO Excel export file (reads from row 7)"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Options")
        
        if st.button("🚀 Start Reconciliation", type="primary", use_container_width=True):
            if calypso_file is None or insto_file is None:
                st.error("⚠️ Please upload both files first!")
                return
            st.session_state.run_check = True
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("""
        1. Upload Calypso CSV file
        2. Upload instO Excel file  
        3. Click 'Start Reconciliation'
        4. Download the Excel report
        """)
    
    # Main content area
    if 'run_check' in st.session_state and st.session_state.run_check:
        if calypso_file and insto_file:
            # Create progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander("📋 Processing Log", expanded=True)
            
            logs = []
            def log(msg):
                logs.append(msg)
                with log_container:
                    for l in logs:
                        st.text(l)
            
            try:
                # Step 1: Load
                status_text.text("Step 1/8: Loading Calypso data...")
                progress_bar.progress(10)
                df_cal_raw = load_calypso_data(calypso_file.getvalue())
                if df_cal_raw is None:
                    return
                log(f"✓ Loaded Calypso: {df_cal_raw.shape[0]} rows, {df_cal_raw.shape[1]} columns")
                
                status_text.text("Step 2/8: Loading instO data...")
                progress_bar.progress(20)
                df_inst_raw = load_insto_data(insto_file.getvalue())
                if df_inst_raw is None:
                    return
                log(f"✓ Loaded instO: {df_inst_raw.shape[0]} rows, {df_inst_raw.shape[1]} columns")
                
                # Step 2: Filter
                status_text.text("Step 3/8: Filtering instO data...")
                progress_bar.progress(30)
                df_inst_clean = filter_insto_data(df_inst_raw)
                removed = len(df_inst_raw) - len(df_inst_clean)
                log(f"✓ Filtered instO: removed {removed} rows, {len(df_inst_clean)} remaining")
                
                status_text.text("Step 4/8: Filtering Calypso data...")
                progress_bar.progress(40)
                df_cal_clean = filter_calypso_data(df_cal_raw)
                log(f"✓ Filtered Calypso: {len(df_cal_clean)} rows kept")
                
                # Step 3: Mapping
                status_text.text("Step 5/8: Adding mappings...")
                progress_bar.progress(50)
                df_inst_clean = add_book_mapping(df_inst_clean)
                df_cal_clean = add_simplified_isin(df_cal_clean)
                log("✓ Mappings added")
                
                # Step 4: Split & Merge
                status_text.text("Step 6/8: Splitting by ISIN...")
                progress_bar.progress(60)
                df_inst_with_isin, df_inst_without_isin = split_insto_by_isin(df_inst_clean)
                log(f"✓ Split: {len(df_inst_with_isin)} with ISIN, {len(df_inst_without_isin)} without ISIN")
                
                status_text.text("Step 7/8: Merging data...")
                progress_bar.progress(70)
                merged_with_isin = merge_with_isin(df_inst_with_isin, df_cal_clean)
                merged_without_isin = merge_without_isin(df_inst_without_isin, df_cal_clean)
                df_merged = combine_merged_results(merged_with_isin, merged_without_isin)
                matched = df_merged['Calypso Quantity'].notna().sum()
                log(f"✓ Merged: {matched}/{len(df_merged)} matched ({matched/len(df_merged)*100:.1f}%)")
                
                # Step 5: Analyze
                status_text.text("Step 8/8: Analyzing differences...")
                progress_bar.progress(85)
                df_result = analyze_quantity_matches(df_merged)
                
                # Count discrepancies
                errors = df_result[df_result['var'].apply(lambda x: isinstance(x, (int, float)) and abs(x) >= 0.001)]
                log(f"✓ Analysis complete: {len(errors)} discrepancies found")
                
                # Generate report
                progress_bar.progress(95)
                b64, df_export = generate_excel_download(df_result)
                
                # Display results
                progress_bar.progress(100)
                status_text.text("✅ Reconciliation Complete!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(df_result))
                with col2:
                    matched_count = len(df_export[df_export['Status'] == 'Match'])
                    st.metric("Matched", matched_count, f"{matched_count/len(df_result)*100:.1f}%")
                with col3:
                    mismatched = len(df_export[df_export['Status'] == 'Mismatch'])
                    st.metric("Mismatched", mismatched, f"{mismatched/len(df_result)*100:.1f}%")
                with col4:
                    repos = len(df_export[df_export['var'] == 'REPO'])
                    st.metric("REPO Cases", repos)
                
                # Data preview
                st.subheader("📋 Reconciliation Results Preview")

                # 修复：将var列转为字符串，避免Arrow序列化错误
                df_display = df_export.copy()
                df_display['var'] = df_display['var'].astype(str)

                # 同时修复 use_container_width 警告（改为 width='stretch'）
                st.dataframe(df_display, width='stretch', height=400)
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_check_results_{timestamp}.xlsx"
                
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;font-size:16px;">📥 Download Excel Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                
                if mismatched > 0:
                    with st.expander("🔍 View Discrepancies Only", expanded=False):
                        mismatched_df = df_export[df_export['Status'] == 'Mismatch'].copy()
                        mismatched_df['var'] = mismatched_df['var'].astype(str)
                        st.dataframe(mismatched_df, width='stretch')
                
                # Clear run flag
                st.session_state.run_check = False
                
            except Exception as e:
                st.error(f"❌ Error during reconciliation: {str(e)}")
                st.exception(e)
                st.session_state.run_check = False
    else:
        # Show welcome message
        st.info("👈 Please upload files in the sidebar and click 'Start Reconciliation'")

if __name__ == "__main__":
    main()