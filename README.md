# Customer_Churn_Prediction_Webapp
"A Machine Learning Web App built with Streamlit that predicts whether a telecom customer will leave (churn) or stay using the Random Forest algorithm. It analyzes factors like tenure, monthly charges, and internet service type to identify at-risk customers."
Here is a Detailed Project Description in English. You can use this for your GitHub README.md, your CV/Resume, or a project report.

🚀 End-to-End Telco Customer Churn Prediction App
1. Project Overview
Customer Churn (when a customer stops using a service) is a major problem for telecom companies. The goal of this project was to build a Machine Learning solution that can predict whether a specific customer is likely to leave or stay based on their profile. We deployed this model as a user-friendly Web Application using Streamlit, allowing non-technical users to input data and get real-time risk assessments.

2. Workflow & Methodology
The project was executed in two major phases: Model Training and Web App Development.

Phase 1: Machine Learning Pipeline (Backend)
Data Collection: We used the standard "Telco Customer Churn" dataset containing 7,000+ customer records.

Data Cleaning:

Handled missing values in the TotalCharges column.

Converted the target variable (Churn) into binary format (1 for Yes, 0 for No).

Feature Engineering:

Applied One-Hot Encoding (pd.get_dummies) to convert categorical text data (e.g., Gender, Internet Service, Payment Method) into numeric format so the model could understand it.

Model Selection:

We chose the Random Forest Classifier because it provides high accuracy and handles a mix of numerical and categorical features effectively.

The model achieved an accuracy of approximately 78.5% on the test data.

Explainability (XAI):

We used SHAP (SHapley Additive exPlanations) to understand the "Why" behind predictions. We discovered that High Monthly Charges and Fiber Optic Internet were the top factors driving churn.

Model Serialization:

The trained model was saved using joblib (churn_model.pkl).

Crucially, we also saved the list of column names (model_columns.pkl) to ensure the web app processes inputs exactly like the training data.

Phase 2: Web Application Development (Frontend)
We built an interactive web interface using Streamlit to make the model accessible.

User Interface (UI):

Designed a Sidebar using st.sidebar containing sliders for numerical inputs (e.g., Tenure, Monthly Charges) and dropdown menus for categorical inputs (e.g., Contract Type, Payment Method).

Data Processing Pipeline in the App:

Input Capture: The app captures user inputs and converts them into a Pandas DataFrame.

Encoding: It applies pd.get_dummies() to the user input.

Schema Alignment: Critical Technical Step—Since a single user might not select every category, the input DataFrame often has fewer columns than the trained model requires. We used .reindex(columns=model_columns, fill_value=0) to align the input data structure perfectly with the trained model.

Prediction:

The app loads the saved "Brain" (churn_model.pkl) and calculates the probability of churn.

Output:

If the probability is high, it displays a "High Risk" warning.

If the probability is low, it displays a "Safe" success message.

3. Key Insights & Results
Accuracy: The model correctly predicts customer behavior ~78% of the time.

Business Insight: Customers on Month-to-Month contracts with high monthly bills are the most likely to leave. Focusing retention offers on this group could significantly reduce losses.

4. Tech Stack
Programming Language: Python

Machine Learning: Scikit-Learn (Random Forest)

Data Manipulation: Pandas, NumPy

Model Interpretability: SHAP

Web Framework: Streamlit

Model Deployment: Joblib
