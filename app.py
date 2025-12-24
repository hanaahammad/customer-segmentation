import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import linkage
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION AND INITIAL DATA LOAD ---

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="CPMAI Customer Segmentation Project",
    layout="wide",
    # Sidebar is collapsed by default to focus on the introduction and loading status
    initial_sidebar_state="collapsed"
)

# --- Initialize Session State for Loading Control ---
# This controls whether the heavy modeling function has been called.
if 'models_ready' not in st.session_state:
    st.session_state['models_ready'] = False

# --- CACHE RAW DATA LOAD ONLY (FAST) ---
@st.cache_data
def load_raw_data():
    """
    Loads and caches the entire raw dataset.
    This function is run only once, greatly speeding up subsequent app loads.
    """
    try:
        # NOTE: Assumes 'OnlineRetail.csv' is in the root directory
        df = pd.read_csv('OnlineRetail.csv.gz',  compression='gzip', encoding='latin-1')
        #pd.read_csv("data/data.csv.gz", compression="gzip")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        return df
    except FileNotFoundError:
        st.error("Error: OnlineRetail.csv not found. Please place the file in the project directory.")
        return pd.DataFrame()

# --- 2. CORE DATA PROCESSING AND MODELING FUNCTION (OPTIMIZED) ---

@st.cache_data
def generate_models(df):
    """
    Performs data cleaning, RFM calculation, model training, and evaluation.
    This heavy computation is cached and runs only once until the input 'df' changes.
    """
    if df.empty:
        # Returns: rfm_df, enriched_df, metrics_df, hierarchical_linkage, dbscan_labels
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None 

    # --- Aggressive Customer Sampling for Speed (CPMAI Phase 3 Optimization) ---
    max_date = df['InvoiceDate'].max()
    start_date = max_date - pd.DateOffset(years=1)
    df_recent = df[df['InvoiceDate'] >= start_date].copy()
    
    all_customers = df_recent['CustomerID'].unique()
    sample_size = int(len(all_customers) * 0.20)
    sampled_customers = np.random.choice(all_customers, size=sample_size, replace=False)
    
    df_clean = df_recent[df_recent['CustomerID'].isin(sampled_customers)].copy()
    
    # Final cleaning steps
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.contains('C', na=False)]
    df_clean.dropna(subset=['CustomerID'], inplace=True)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # --- RFM Calculation (CPMAI Phase 2: Feature Engineering) ---
    NOW = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df_clean.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (NOW - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # --- BASELINE MODEL (RFM-ONLY) ---
    
    # Log Transformation and Standard Scaling (CPMAI Phase 3)
    rfm_log = rfm_df[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
    scaler_baseline = StandardScaler()
    rfm_scaled = scaler_baseline.fit_transform(rfm_log)
    
    # Create scaled DF with CustomerID from the original rfm_df for guaranteed alignment
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['R_Scaled', 'F_Scaled', 'M_Scaled'])
    rfm_scaled_df['CustomerID'] = rfm_df['CustomerID'].values 
    
    X_scaled = rfm_scaled_df[['R_Scaled', 'F_Scaled', 'M_Scaled']].values

    # --- K-Means Experimentation Loop (CPMAI Phase 4/5) ---
    MAX_K = 10
    inertia = []
    silhouette = []
    
    for k in range(2, MAX_K + 1):
        # NOTE: Setting n_init explicitly to 10 to suppress future Scikit-learn warnings
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10) 
        kmeans_model.fit(X_scaled)
        inertia.append(kmeans_model.inertia_)
        score = silhouette_score(X_scaled, kmeans_model.labels_)
        silhouette.append(score)

    evaluation_metrics_df = pd.DataFrame({
        'K': range(2, MAX_K + 1),
        'Inertia': inertia,
        'Silhouette Score': silhouette
    })
    
    # --- Final Baseline Clustering (K=4) ---
    K_baseline = 4 
    kmeans_baseline = KMeans(n_clusters=K_baseline, random_state=42, n_init=10)
    rfm_df['Baseline_Cluster'] = kmeans_baseline.fit_predict(X_scaled)

    # Characterize and map the segments for business interpretability
    cluster_profiles = rfm_df.groupby('Baseline_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    sorted_profiles = cluster_profiles.sort_values(by=['Recency', 'Monetary'], ascending=[True, False]).index.tolist()
    segment_map = {
        sorted_profiles[0]: 'Champions (Highest Value)', 
        sorted_profiles[1]: 'Loyal Customers',
        sorted_profiles[2]: 'At-Risk (Fading)',
        sorted_profiles[3]: 'Lost (Hibernating)' 
    }
    rfm_df['Baseline_Segment'] = rfm_df['Baseline_Cluster'].map(segment_map)
    
    # NEW STEP: Merge the scaled features back into the main rfm_df for Page 6 visualization
    rfm_df = rfm_df.merge(rfm_scaled_df, on='CustomerID', how='left') 

    # --- Comparison Algorithm Generation (For Page 7) ---
    hierarchical_linkage = linkage(X_scaled, method='ward')
    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.fit_predict(X_scaled)
    rfm_df['DBSCAN_Label'] = dbscan_labels
    
    # --- ENRICHED MODEL (RFM + CATEGORICAL) ---
    rfm_enriched_df = rfm_df.copy() # Use the now-enriched rfm_df as a base

    # Feature Engineering for enrichment
    df_clean['Category'] = df_clean['Description'].astype(str).apply(lambda x: x.split(' ')[0].upper())
    category_spend = df_clean.groupby(['CustomerID', 'Category'])['TotalPrice'].sum().reset_index()
    max_spend_category = category_spend.loc[category_spend.groupby('CustomerID')['TotalPrice'].idxmax()].rename(columns={'Category': 'Top_Category'})[['CustomerID', 'Top_Category']]
    customer_country = df_clean.groupby('CustomerID')['Country'].apply(lambda x: x.mode()[0]).reset_index()

    rfm_enriched_df = rfm_enriched_df.merge(max_spend_category, on='CustomerID', how='left')
    rfm_enriched_df = rfm_enriched_df.merge(customer_country, on='CustomerID', how='left')
    rfm_enriched_df['Top_Category'].fillna('OTHER', inplace=True)

    # Grouping low-frequency categories and countries
    top_countries = rfm_enriched_df['Country'].value_counts().head(5).index.tolist()
    top_categories = rfm_enriched_df['Top_Category'].value_counts().head(10).index.tolist()
    rfm_enriched_df['Country_Grouped'] = np.where(rfm_enriched_df['Country'].isin(top_countries), rfm_enriched_df['Country'], 'OTHER_COUNTRY')
    rfm_enriched_df['Category_Grouped'] = np.where(rfm_enriched_df['Top_Category'].isin(top_categories), rfm_enriched_df['Top_Category'], 'OTHER_CATEGORY')

    # Preprocessing Pipeline for Enriched Model
    numerical_features = ['Recency', 'Frequency', 'Monetary']
    categorical_features = ['Country_Grouped', 'Category_Grouped']

    # Use ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    X_processed = preprocessor.fit_transform(rfm_enriched_df[['Recency', 'Frequency', 'Monetary', 'Country_Grouped', 'Category_Grouped']])

    # Enriched Clustering (K=6 for more detailed segmentation)
    K_enriched = 6
    kmeans_enriched = KMeans(n_clusters=K_enriched, random_state=42, n_init=10)
    rfm_enriched_df['Enriched_Cluster'] = kmeans_enriched.fit_predict(X_processed)
    rfm_enriched_df['Enriched_Segment'] = 'Group ' + (rfm_enriched_df['Enriched_Cluster'] + 1).astype(str)
    
    # Return the segmented DataFrames, evaluation metrics, and comparison model data
    return rfm_df, rfm_enriched_df, evaluation_metrics_df, hierarchical_linkage, dbscan_labels


# --- 3. EXECUTION AND SESSION STATE MANAGEMENT (WITH UI PROGRESS) ---

raw_df = load_raw_data()

# Only run the status block if data was loaded successfully
if not raw_df.empty:
    
    # --- Introduction Block (Displayed first, requires button click to proceed) ---
    if not st.session_state['models_ready']:
        st.title("🎯 Data-Driven Customer Segmentation with CPMAI")
        st.markdown("---")
        
        st.header("🛒 Source Dataset: Online Retail")
        st.markdown("""
        This project utilizes a publicly available **Online Retail Transactional Dataset**.
        
        * **Nature:** It contains all transactions for a UK-based non-store online retail company over a 12-month period (December 2010 to December 2011).
        * **Key Data Points:** Each row includes critical information such as **`InvoiceNo`**, **`StockCode`**, **`Quantity`**, **`InvoiceDate`**, and the vital **`CustomerID`** and **`Country`**.
        * **Purpose:** This rich transaction history is the foundation for calculating the **RFM (Recency, Frequency, Monetary)** features that drive the customer segmentation model.
        """)
        
        st.header("Why Customer Segmentation Matters for Business")
        st.markdown("""
        Treating all customers the same is inefficient. **Customer Segmentation** uses machine learning (clustering) to group customers based on their **RFM** (Recency, Frequency, Monetary value) behavior, allowing us to:
        
        * **Optimize Marketing:** Send targeted offers only to the most receptive groups (e.g., retention offers to 'At-Risk' customers).
        * **Improve Profitability:** Dedicate high-cost resources to 'Champions' and high-value segments.
        * **Align with Business Goals:** Ensure the clusters are clearly defined and actionable, providing immediate value to sales and marketing teams.
        """)
        
        st.header("The CPMAI Methodology (Cognitive Project Management for AI)")
        st.markdown("""
        To ensure the resulting models meet **business user expectations** and are traceable, we apply the **CPMAI framework**. This enforces structure, guaranteeing that data preparation, modeling choices (like selecting $K=4$), and evaluation are strictly aligned with the initial business goals of maximizing customer lifetime value and minimizing churn.
        """)
        
        st.markdown("---")
        
        # Button to trigger the heavy loading process
        if st.button("🚀 Start Model Building & Load Project Data", type="primary"):
            st.session_state['models_ready'] = True
            st.rerun() # Rerun the app to move into the loading block

    # --- Loading Block (Displayed after button click) ---
    if st.session_state['models_ready']:
        
        st.title("CPMAI Customer Segmentation Project")
        st.subheader("Initializing Models and Data Pipelines...")
        
        with st.status("Running the CPMAI Modeling Workflow", expanded=True) as status:
            
            # Display the steps before calling the cached function
            st.write("✅ **Phase 2/3: Data Preparation** (Cleaning, RFM Calculation, Log Transform, Scaling)")
            st.write("✅ **Phase 4: Baseline Model Training** (Running K-Means for K=2 to K=10 Experiment)")
            st.write("✅ **Phase 4: Comparison Models** (Training Hierarchical and DBSCAN on RFM data)") 
            st.write("✅ **Phase 4: Enriched Model Training** (RFM + Categorical features for K=6)")
            
            # Execute the heavy computation. Streamlit's cache handles this efficiently.
            rfm_baseline_df, rfm_enriched_df, evaluation_metrics_df, hierarchical_linkage, dbscan_labels = generate_models(raw_df)
            
            # Store results in the session state for pages to access instantly
            st.session_state['rfm_baseline_df'] = rfm_baseline_df
            st.session_state['rfm_enriched_df'] = rfm_enriched_df
            st.session_state['evaluation_metrics_df'] = evaluation_metrics_df
            st.session_state['hierarchical_linkage'] = hierarchical_linkage
            st.session_state['dbscan_labels'] = dbscan_labels


            # Update the status to complete
            status.update(label="Models Trained and Data Loaded! You are ready to navigate.", 
                          state="complete", 
                          expanded=False)
            
            st.success("The project data and models are loaded into memory.")

            st.markdown("**Instructions:** Please expand the sidebar (top-left ☰) to begin the guided tour through the 8 CPMAI phases.")

