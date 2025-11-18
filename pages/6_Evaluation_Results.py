import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def get_segment_descriptions(model_choice):
    """
    Returns business-friendly descriptions for each segment based on the model choice.
    """
    if model_choice == "Baseline Model (RFM-Only)":
        return {
            'Champions (Highest Value)': {
                'icon': '👑', 
                'desc': "These customers are the **best segment**: highly recent, frequent buyers, and spend the most. They are the most valuable and should be rewarded, nurtured, and prioritized for retention and advocacy programs."
            },
            'Loyal Customers': {
                'icon': '💛', 
                'desc': "These are frequent, valuable buyers who form the **backbone** of the business. They respond well to consistent communication and reliable product availability. Focus on rewarding consistency and cross-selling."
            },
            'At-Risk (Fading)': {
                'icon': '⚠️', 
                'desc': "These customers used to be loyal but haven't purchased recently (**Recency is growing**). They require immediate intervention, such as personalized win-back offers or direct customer service follow-ups, to prevent imminent churn."
            },
            'Lost (Hibernating)': {
                'icon': '😴', 
                'desc': "These customers have a very high Recency (**haven't bought in a long time**) and generally low Frequency and Monetary value. They should be de-prioritized for expensive marketing but included in seasonal, low-cost reactivation campaigns."
            }
        }
    else: # Enriched Model (K=6)
        return {
            'Group 1': {'icon': '💎', 'desc': 'Highly engaged, high-spending segment, likely characterized by a specific country or product affinity. Treat as VIPs.'},
            'Group 2': {'icon': '🗺️', 'desc': 'Moderately valuable customers showing regional or categorical purchase similarity. Good targets for cross-selling relevant items.'},
            'Group 3': {'icon': '🚀', 'desc': 'Newly acquired customers showing similar initial purchasing patterns. Focus on detailed onboarding and establishing long-term loyalty.'},
            'Group 4': {'icon': '🎯', 'desc': 'Customers that favor a specific, high-margin product category. Excellent targets for specialized upsell campaigns focused on premium versions of their favorite items.'},
            'Group 5': {'icon': '🕰️', 'desc': 'Customers showing signs of disengagement (high Recency) but who historically bought a specific product type. High-priority for targeted win-back campaigns based on past preference.'},
            'Group 6': {'icon': '🌱', 'desc': 'Low-value, infrequent buyers, likely purchasing non-core items. Nurture them with low-cost digital marketing and look for early indicators of growth.'}
        }

def plot_evaluation_metrics(df_metrics):
    """Plots Inertia (Elbow Method) and Silhouette Score."""
    
    st.subheader("Model Fit Assessment: Finding Optimal $K$")
    st.markdown("We use internal clustering metrics to decide the best number of segments ($K$) for the Baseline Model.")
    
    col1, col2 = st.columns(2)

    # 1. Elbow Method (Inertia)
    with col1:
        st.markdown("#### 1. Elbow Method (Inertia)")
        fig_elbow = px.line(df_metrics, x='K', y='Inertia', 
                            title='Inertia (Within-Cluster Sum of Squares) vs. K', 
                            markers=True)
        fig_elbow.add_vline(x=4, line_width=2, line_dash="dash", line_color="red", 
                            annotation_text="Chosen K=4", annotation_position="top right")
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown(r"""
        **Explanation:** Inertia measures how tight the clusters are. We look for the "elbow"—the point where the decrease in inertia starts to level off. **$K=4$** is selected, balancing low inertia with model simplicity.
        """)

    # 2. Silhouette Score
    with col2:
        st.markdown("#### 2. Silhouette Score")
        fig_silhouette = px.line(df_metrics, x='K', y='Silhouette Score', 
                                 title='Silhouette Score vs. K', 
                                 markers=True)
        fig_silhouette.add_vline(x=4, line_width=2, line_dash="dash", line_color="red", 
                                 annotation_text="Chosen K=4", annotation_position="top right")
        st.plotly_chart(fig_silhouette, use_container_width=True)
        st.markdown(r"""
        **Explanation:** The Silhouette Score ranges from -1 to +1, indicating how well-separated and dense the clusters are. A high, positive score confirms **well-defined clusters**.
        """)
        
    st.markdown("---")


def page_evaluation_results(rfm_baseline_df, rfm_enriched_df, evaluation_metrics_df):
    """CPMAI Phases 4-5 Modeling and Evaluation."""
    st.title("📊 6. Evaluation & Results (CPMAI 4 & 5)")
    st.markdown("---")

    # --- 1. Model Evaluation Metrics ---
    st.header("1️⃣ Technical Evaluation (Baseline Model)")
    plot_evaluation_metrics(evaluation_metrics_df)

    # --- 2. Model Selection and Display ---
    st.header("2️⃣ Cluster Profiling and Business Actionability")

    # The user selects the model on this page for quick comparison
    model_choice = st.selectbox(
        "**Select Model for Analysis:**",
        ["Baseline Model (RFM-Only)", "Enriched Model (RFM + Categorical)"]
    )

    if model_choice == "Baseline Model (RFM-Only)":
        df = rfm_baseline_df
        segment_col = 'Baseline_Segment'
        scaled_cols = ['R_Scaled', 'F_Scaled', 'M_Scaled']
        st.subheader(f"📊 Displaying: {model_choice} (K=4)")
        
    else:
        df = rfm_enriched_df
        segment_col = 'Enriched_Segment'
        # Although the enriched model uses more features, we still plot only RFM scaled features 
        # for a simple, comparable radar chart visualization.
        scaled_cols = ['R_Scaled', 'F_Scaled', 'M_Scaled'] 
        st.subheader(f"📊 Displaying: {model_choice} (K=6)")

    # --- 2.1 Segment Profile Visualization (Radar Chart) ---
    st.subheader("Segment Profile Comparison (Stats and Graph)")
    
    # Calculate RAW mean for table (for business context)
    segment_raw_means = df.groupby(segment_col)[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

    # Calculate SCALED mean for radar chart (for visualization)
    segment_scaled_means = df.groupby(segment_col)[scaled_cols].mean().reset_index()
    # Rename columns for clear axis labels in the plot
    segment_scaled_means.columns = [segment_col, 'Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)'] 

    st.markdown("#### Cluster Mean RFM Statistics (Raw Values)")
    st.dataframe(segment_raw_means.set_index(segment_col)) # Display raw means for context

    st.markdown("#### Segment Profile Radar Chart (Standardized Scores)")
    st.info("To compare Recency, Frequency, and Monetary on the same chart, we plot their **Standardized Mean Scores** (Z-scores). A score of 0 is average; positive means above average for that metric; negative means below average. **This fixes the scale issue!**")

    # Melt the Scaled DataFrame from wide to long format for the polar plot
    segment_scaled_long = segment_scaled_means.melt(
        id_vars=[segment_col],
        value_vars=['Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)'],
        var_name='Metric',
        value_name='Value'
    )
    
    # Set the range symmetrically around zero for a clear visual comparison of Z-scores
    min_val = segment_scaled_long['Value'].min()
    max_val = segment_scaled_long['Value'].max()
    max_abs = max(abs(min_val), abs(max_val))
    
    # Ensure a small buffer
    range_min = -max_abs * 1.1
    range_max = max_abs * 1.1
    
    # If, somehow, all values are positive (unlikely for Z-scores) ensure 0 is the min
    if range_min > 0:
        range_min = 0 


    fig = px.line_polar(
        segment_scaled_long, # Use the scaled long format
        r='Value',
        theta='Metric',
        color=segment_col,
        line_close=True,
        height=550
    )
    # Update radial axis to show range symmetrically, which is key for scaled features
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[range_min, range_max])))
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW SECTION: Business Description ---
    st.markdown("---")
    st.header("3️⃣ Business Segment Descriptions")
    st.markdown(f"A summary profile for each of the {len(df[segment_col].unique())} segments, detailing their behavioral characteristics and value to the business.")

    segment_descriptions = get_segment_descriptions(model_choice)
    
    # Display descriptions in a clean column layout
    cols = st.columns(len(df[segment_col].unique()))
    
    # Ensure segments are iterated in the correct, present order from the dataframe
    sorted_segments = segment_raw_means[segment_col].tolist()
    
    for i, seg_name in enumerate(sorted_segments):
        # Fallback to a default if a segment name isn't found in the map (shouldn't happen)
        desc_data = segment_descriptions.get(seg_name, {'icon': '❓', 'desc': 'No specific description available.'})
        
        with cols[i % len(cols)]: # Cycle through columns if more than 4-6 segments
            st.markdown(f"#### {desc_data['icon']} {seg_name}")
            st.markdown(desc_data['desc'])


    # --- 4. Business Evaluation and Action Plan ---
    st.markdown("---")
    st.header("4️⃣ Business Actionability")
    
    # (The existing action plan logic follows)
    
    action_df = segment_raw_means.copy()
    action_df['RFM Profile'] = action_df.apply(lambda row: f"R: {row['Recency']:.0f} | F: {row['Frequency']:.1f} | M: ${row['Monetary']:.0f}", axis=1)
    
    if model_choice == "Baseline Model (RFM-Only)":
        # K=4 segments
        action_df['Business Action'] = [
            'Reward/VIP Program, Solicit Referrals', 
            'Cross-Sell/Upsell based on purchase history', 
            'Win-Back Offers, Customer Service Intervention', 
            'De-prioritize or Run a Seasonal Reactivation Campaign'
        ]
    else:
        # K=6 segments - actions remain generic as before
        action_df['Business Action'] = [
            'Specialized Offer A (High Value)', 
            'Geo-Targeted Retention B', 
            'New Customer Onboarding Flow C', 
            'Product Focus D (High Margin)', 
            'At-Risk Intervention E',
            'Low-Value Nurturing F' 
        ]

    st.table(action_df[[segment_col, 'RFM Profile', 'Business Action']].set_index(segment_col))


# --- Check Session State ---
if 'rfm_baseline_df' in st.session_state and 'evaluation_metrics_df' in st.session_state:
    # --- CORRECTION APPLIED HERE: st.session_session_state changed to st.session_state ---
    page_evaluation_results(st.session_state['rfm_baseline_df'], st.session_state['rfm_enriched_df'], st.session_state['evaluation_metrics_df'])
else:
    st.warning("Please wait for data and models to load.")