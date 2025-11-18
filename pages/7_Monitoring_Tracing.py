import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def simulate_new_data(rfm_train_df):
    """
    Simulates a new batch of production data with intentional drift for demonstration.
    The drift scenario: customers are becoming less recent (Recency increases)
    and spending slightly less (Monetary decreases).
    """
    st.subheader("Simulating Production Data Drift")
    with st.expander("Drift Simulation Parameters", expanded=False):
        
        # --- Drift Parameters ---
        recency_drift_days = st.slider("Recency Drift (Days added)", min_value=0, max_value=30, value=7)
        monetary_drift_factor = st.slider("Monetary Drift (Factor)", min_value=0.8, max_value=1.0, value=0.95, step=0.01)
        
    df_new = rfm_train_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # 1. Recency Drift (Customers are taking longer to return)
    df_new['Recency'] = df_new['Recency'] + recency_drift_days
    
    # 2. Monetary Drift (Customers are spending slightly less)
    df_new['Monetary'] = df_new['Monetary'] * monetary_drift_factor
    
    # 3. Frequency remains relatively stable but slightly varied (minimal change)
    df_new['Frequency'] = df_new['Frequency'] + np.random.randint(-1, 2, size=len(df_new))
    df_new['Frequency'] = df_new['Frequency'].apply(lambda x: max(1, x)) # Ensure frequency is at least 1
    
    df_new['DataType'] = 'New Production Data'
    rfm_train_df['DataType'] = 'Training Data'
    
    return df_new, rfm_train_df[['Recency', 'Frequency', 'Monetary', 'DataType']]

def plot_distribution_comparison(df_combined, feature_name):
    """Plots the distribution comparison between training and new data for a given feature."""
    
    # We use a combined dataset for a unified plot.
    fig = px.histogram(
        df_combined,
        x=feature_name,
        color='DataType',
        barmode='overlay',
        histnorm='probability density',
        opacity=0.6,
        title=f'Distribution Comparison: {feature_name}',
    )
    fig.update_layout(height=400, legend_title_text='Data Source')
    
    return fig

def page_deployment_monitoring(rfm_baseline_df):
    """CPMAI Phase 6: Deployment and Monitoring."""
    st.title("🚨 8. Deployment & Monitoring (CPMAI Phase 6)")
    st.markdown("---")
    
    st.header("1️⃣ The Monitoring Challenge: Unsupervised Drift")
    st.markdown("""
    For a clustering model, traditional **Model Drift** (like accuracy decline in classification) doesn't apply. Instead, we monitor **Data Drift**—changes in the input features (**RFM**) over time—which leads to **Cluster Validity Degradation**.
    
    If the new customer data distribution significantly shifts from the training data, the original cluster centroids become invalid, and the model must be retrained.
    """)

    # --- Data Simulation and Combination ---
    df_train_only_rfm = rfm_baseline_df[['Recency', 'Frequency', 'Monetary']]
    df_new_simulated, df_train_tagged = simulate_new_data(df_train_only_rfm)
    
    df_combined = pd.concat([df_train_tagged, df_new_simulated], ignore_index=True)
    
    # --- Visualization ---
    st.header("2️⃣ Monitoring Data Distribution Drift (RFM Features)")
    st.markdown("""
    Below, we compare the distribution of the three core RFM features from the **Training Data** (used to set the cluster centers) versus the **Simulated New Production Data**. Significant deviation (drift) indicates that the existing K-Means model is no longer valid.
    """)
    
    # Plot comparisons for all three features
    col_r, col_f, col_m = st.columns(3)
    
    with col_r:
        fig_r = plot_distribution_comparison(df_combined, 'Recency')
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("**Observation:** The shift to the right indicates customers are taking longer between purchases (Recency Drift).")
        
    with col_f:
        fig_f = plot_distribution_comparison(df_combined, 'Frequency')
        st.plotly_chart(fig_f, use_container_width=True)
        st.markdown("**Observation:** Minimal shift, suggesting transaction count remains relatively stable.")
        
    with col_m:
        fig_m = plot_distribution_comparison(df_combined, 'Monetary')
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown("**Observation:** Slight shift to the left, indicating a marginal decrease in average customer spend (Monetary Drift).")

    st.markdown("---")
    
    st.header("3️⃣ Action Plan for Drift Detection")
    st.markdown("""
    When data drift is detected (e.g., the Recency distribution shifts by a statistically significant amount for 3 consecutive monitoring periods), the following steps are triggered:
    
    1.  **Alert:** An automatic alert is sent to the Data Science and Marketing teams.
    2.  **Validation:** Human review validates the severity and cause of the drift.
    3.  **Retraining:** The entire CPMAI workflow (Data Preparation, Modeling, Evaluation) is executed on the **newest 12 months of production data**.
    4.  **Deployment:** The new, validated K-Means model is deployed, and the old cluster centroids are retired.
    """)


# --- Check Session State ---
if 'rfm_baseline_df' in st.session_state:
    page_deployment_monitoring(st.session_state['rfm_baseline_df'])
else:
    st.title("Loading Project Data...")
    st.warning("Please wait for the main application (app.py) to finish loading the data and models.")