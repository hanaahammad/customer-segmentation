import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import pairwise_distances_argmin_min

# --- 1. Synthetic Data Generation ---

@st.cache_data
def generate_simple_data():
    """Generates simple 2D data with 3 distinct clusters."""
    # X holds the coordinates, y holds the true cluster labels
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    df = pd.DataFrame(X, columns=['Feature_X', 'Feature_Y'])
    df['True_Cluster'] = y
    return df

# --- 2. Original Model Training ---

@st.cache_data
def train_original_model(df):
    """Trains K-Means on the original data to establish baseline centroids."""
    # We train K-Means on the original, clean data
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(df[['Feature_X', 'Feature_Y']])
    return kmeans.cluster_centers_

# --- 3. Drift Simulation and Decayed Prediction ---

def get_decayed_clusters(df_original, original_centroids, drift_amount):
    """
    Applies data drift and assigns clusters using the OLD (decayed) centroids.
    This simulates production where the model hasn't been retrained.
    """
    df_drifted = df_original.copy()
    
    # 1. Apply data drift (shift the Recency/Feature_X distribution)
    df_drifted['Feature_X'] = df_drifted['Feature_X'] + drift_amount 
    
    # 2. Predict clusters on the drifted data using the ORIGINAL centroids
    # This simulates the decaying model in production
    decayed_labels, _ = pairwise_distances_argmin_min(
        df_drifted[['Feature_X', 'Feature_Y']].values,
        original_centroids
    )
    df_drifted['Decayed_Cluster'] = decayed_labels
    
    return df_drifted

# --- 4. Plotting Function ---

def plot_clusters(df_data, centroids, cluster_col, title):
    """Generates a scatter plot with cluster assignments and centroid markers."""
    fig = px.scatter(
        df_data, 
        x='Feature_X', 
        y='Feature_Y', 
        color=cluster_col, 
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_white'
    )
    
    # Add centroids (cluster centers) as large stars
    centroids_df = pd.DataFrame(centroids, columns=['Feature_X', 'Feature_Y'])
    fig.add_scatter(
        x=centroids_df['Feature_X'], 
        y=centroids_df['Feature_Y'], 
        mode='markers', 
        marker=dict(symbol='star', size=20, color='black', line=dict(width=2, color='white')),
        name='Centroids (Original Model)'
    )
    
    fig.update_layout(height=500, xaxis_title="Feature X (e.g., Recency)", yaxis_title="Feature Y (e.g., Monetary)")
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='lightgray')))
    
    return fig

# --- 5. Streamlit Page ---

def page_clustering_explained():
    st.title("💡 9. Clustering Explained Simply: The Drift Problem")
    st.markdown("---")
    
    st.header("1️⃣ The Setup: Training the Model")
    st.markdown("""
    Imagine a simple customer segmentation based on two features (X and Y). The **Original Model** (K-Means) is trained on the data below and establishes **3 Cluster Centroids** (the black stars).
    """)

    df_original = generate_simple_data()
    original_centroids = train_original_model(df_original)

    st.header("2️⃣ Simulation Control: Data Drift")
    
    col_a, col_b = st.columns([1, 2])

    with col_a:
        drift_amount = st.slider(
            "Simulate Data Drift (Shift in Feature X)", 
            min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
            help="This simulates a feature distribution change, e.g., if Recency values suddenly increased by 2 days across all customers."
        )
        st.metric("Drift Amount Applied to X", f"{drift_amount:.1f} Units")

    with col_b:
        st.warning("Drag the slider to the right. Observe how the data points shift, but the **Original Centroids** stay in place, causing misclassification of the new data.")
        
    st.markdown("---")

    # Get the drifted data and use the old centroids to assign decayed clusters
    df_drifted_with_decay = get_decayed_clusters(df_original, original_centroids, drift_amount)

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.subheader("Initial State (Drift = 0.0)")
        fig_original = plot_clusters(df_original, original_centroids, 'True_Cluster', 'Original Clusters (Training Data)')
        st.plotly_chart(fig_original, use_container_width=True)
        st.markdown("When the drift is zero, the model works perfectly. Each customer is correctly assigned to its intended segment.")

    with col_viz2:
        st.subheader(f"Model Decay: Applying Original Model to Shifted Data")
        fig_decayed = plot_clusters(df_drifted_with_decay, original_centroids, 'Decayed_Cluster', 'Decayed Cluster Assignment (Production Data)')
        st.plotly_chart(fig_decayed, use_container_width=True)
        st.markdown(f"At a drift of **+{drift_amount:.1f}**, the original black centroids no longer align with the shifted data groups. Segments are assigned incorrectly, leading to failed marketing actions.")

    st.header("3️⃣ Conclusion: Why Retraining is Necessary")
    st.markdown("""
    The visualization above demonstrates **Cluster Validity Degradation**. When the underlying data distribution changes (Data Drift), the mathematical centers of the original clusters (centroids) become inaccurate.
    
    * **Actionable Monitoring:** This is why continuous monitoring (as shown on Page 8) is required. Once drift is confirmed, the model must be **retrained** on the new data to find the *correct* centers.
    """)

# --- Check Session State ---
# This page is simple and doesn't strictly need the main app to finish, but we check for consistency.
if 'models_ready' in st.session_state:
    page_clustering_explained()
else:
    st.title("Loading Project Data...")
    st.warning("Please wait for the main application (app.py) to finish loading the data and models.")