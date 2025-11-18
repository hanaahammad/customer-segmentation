# 📊 CPMAI Customer Segmentation Project: RFM & Model Evolution

This project demonstrates a complete customer segmentation workflow following the six phases of the **CPMAI (Cognitive Project Management for AI)** framework. It uses K-Means clustering on **RFM (Recency, Frequency, Monetary)** features derived from an Online Retail dataset to identify distinct, actionable customer segments.

The dashboard, built with Streamlit, provides a phase-by-phase review, including data preparation, model evaluation, algorithm comparison, and a final Deployment & Monitoring plan.

### 💡 Project Purpose: Engaging Business Stakeholders
This dashboard is designed as a direct bridge between the data science team and business stakeholders. By following the structured CPMAI framework, we aim to achieve full transparency and manage expectations throughout the AI lifecycle. This ensures the resulting models meet real-world needs and are actively maintained.

This guided approach is crucial because it defines the role of business users—not just as consumers of the output, but as active participants:

1.  **Evaluation Intervention:**  Business users validate the model by confirming if the generated customer segments (e.g., 'Champions' vs. 'At-Risk') are actionable and align with real-world marketing strategies (CPMAI Phase 5).

2.  **Monitoring Expectations:** We explicitly set the expectation that customer behavior will drift over time. This requires business users to monitor the Data Distribution Drift (CPMAI Phase 6) and provide essential context to determine when model retraining is necessary, thus ensuring the AI solution maintains its value.

![alt text](https://view.genially.com/68d718770f6db2cb685dc64c/horizontal-infographic-diagrams-cpmai-in-order)



## 🚀 Project Goal

The primary objective is to develop actionable customer segments for targeted marketing strategies. We show the progression from a simple, interpretable **RFM-only Baseline Model** to a more granular **Enriched Model** that incorporates demographic and behavioral features (Country and Top Product Category).

## 🗺️ CPMAI Methodology Covered

The project is structured to address the needs of business users across all 6 CPMAI phases:

1.  **Business Understanding:** Define segments (e.g., 'Champions', 'At-Risk').
2.  **Data Understanding & Preparation:** RFM calculation, cleaning, scaling, and feature enrichment.
3.  **Modeling:** Comparison of K-Means, DBSCAN, and Hierarchical (demonstrated via K-Means).
4.  **Evaluation:** Interactive comparison of Baseline vs. Enriched models, highlighting the business user's role in judging **Actionability**.
5.  **Operationalization**: Or Deployment & Monitoring via a  Dashboard view simulating continuous tracking to detect **Concept Drift**.

## 💻 Tech Stack

* **Language:** Python 3.11+ (Alternative) - Required to run the application locally.
* **Libraries:** `pandas`, `scikit-learn`, `plotly`
* **Web Framework:** Streamlit
* **Environment:** Docker (for reproducibility)
* **Dataset:** Kaggle Online Retail Dataset (`OnlineRetail.csv`)

## 🛠️ Setup and Execution

### Option 1: Using Docker (Recommended for Peer Review)

Ensure you have Docker installed and running.

1.  **Clone the repository / Place files in one folder:**
    ```bash
    git clone [YOUR-REPO-LINK]
    cd customer_segmentation
    ```
2.  **Build the Docker image:**
    - First, ensure your requirements.txt file is present (containing streamlit, pandas, numpy, scikit-learn, plotly) and then build the image:
    ```bash
    docker build -t rfm-segmentation-app .
    ```
3.  **Run the container:**
    - Run the image, mapping port 8501 and mounting the data volume (OnlineRetail.csv) from your current directory ($(pwd)) to the container's working directory (/app):
    - Ensure you are inside the 'customer-segmentation-app' directory with OnlineRetail.csv present.

    ```bash
    docker run -p 8501:8501 rfm-segmentation-app
    ```
4.  **Access the app:** Open your web browser to `http://localhost:8501`.

### Option 2: Local Environment Setup

1.  **Install dependencies:**
    ```bash
    # Create and activate environment first (optional but recommended)
    python -m venv venv
    source venv/bin/activate
    
    # Install Python packages
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
3.  **Access the app:** The terminal will provide the local URL, typically `http://localhost:8501`.

## 💡 Application Usage Guide
When you navigate to the application (http://localhost:8501), the model building process is not yet complete.
1. Start Data Loading and Model Development
The home screen (app.py) will display an introduction and a button:
•	Click the "🚀 **Start Model Building & Load Project Data"** button.
This action triggers the core CPMAI Phase 3 & 4 workflow: data cleaning, RFM feature engineering, and the training of the Baseline (K=4) and Enriched (K=6) clustering models. A status bar will display progress.
2. Navigate the CPMAI Phases



| Page No. | CPMAI Phase | Focus |
| :---: | :---: | :--- |
| **1-2** | Phase 1 (Understanding) | Project Goals, Success Criteria, and AI Pattern. |
| **3-4** | Phase 2/3 (Data) | Data Cleaning, RFM Feature Engineering, and Transformation. |
| **5** | Phase 4 (Modeling) | K-Means Algorithm Selection and Model Definition (Baseline & Enriched). |
| **6** | Phase 5 (Evaluation) |Cluster Profiling, RFM Visualization, and Business Actionability. |
| **7** | Phase 4/5 (Experiment) | Algorithm Comparison (Hierarchical vs. DBSCAN) and Optimal K Selection. |
| **8-9** | Phase 6 (Monitoring) | Monitoring for Data Distribution Drift and Retraining Plan. |
| **9** | Phase 6 (Monitoring) | Technical Explanation of Cluster Decay and Drift Simulation. |




---

## 📁 Project File Structure
```plaintext
customer_segmentation/ 
├── app.py # Main entry point, data cleaning, caching, and all model training. 
├── OnlineRetail.csv # Source data file used for the project. 
├── requirements.txt # Python package dependencies (pandas, streamlit, sklearn, etc.). 
└── pages/ # Contains all Streamlit sidebar pages (The 8-Step CPMAI flow). 
    ├── 1_CPMAI_Introduction.py # Defines the CPMAI framework and Unsupervised ML approach. 
    ├── 2_Business_Understanding.py # CPMAI Phase 1: Problem statement and AI mapping. 
    ├── 3_Data_Understanding.py # CPMAI Phase 2: Data loading, RFM distributions, and feature analysis. 
    ├── 4_Data_Preparation.py # CPMAI Phase 3: Log-transformation and scaling. 
    ├── 5_Algorithm_Choice.py # CPMAI Phase 4 (Theory): Comparison of K-Means, DBSCAN, and Hierarchical. 
    ├── 6_Evaluation_Results.py # CPMAI Phase 5: Cluster profiles, business segmentation, and actionability. 
    ├── 7_Comparison_Algorithms.py # K-Means Experiment (K=2 to 10) results and final algorithm justification. 
    └── 8_Monitoring_Tracing.py # CPMAI Phase 6: Monitoring segment stability and detecting concept drift.


