import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Set page title and configure layout - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# App title and description
st.title("Employee Attrition Prediction")
st.write("Enter employee information to predict if they might leave the job")

# Define all expected columns for the model
def get_expected_columns():
    # The model expects 48 features according to the error
    # Let's make sure all columns from model training are included
    return [
        'Total_Satisfaction_bool', 'Age_bool', 'DailyRate_bool', 'Department_bool',
        'DistanceFromHome_bool', 'JobRole_bool', 'HourlyRate_bool', 'MonthlyIncome_bool',
        'NumCompaniesWorked_bool', 'TotalWorkingYears_bool', 'YearsAtCompany_bool',
        'YearsInCurrentRole_bool', 'YearsSinceLastPromotion_bool', 'YearsWithCurrManager_bool',
        'BusinessTravel_Rarely', 'BusinessTravel_Frequently', 'BusinessTravel_No_Travel',
        'Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5',
        'EducationField_Life_Sciences', 'EducationField_Medical', 'EducationField_Marketing',
        'EducationField_Technical_Degree', 'Education_Human_Resources', 'Education_Other',
        'Gender_Male', 'Gender_Female', 'MaritalStatus_Married', 'MaritalStatus_Single',
        'MaritalStatus_Divorced', 'OverTime_Yes', 'OverTime_No', 'StockOptionLevel_0',
        'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
        'TrainingTimesLastYear_0', 'TrainingTimesLastYear_1', 'TrainingTimesLastYear_2',
        'TrainingTimesLastYear_3', 'TrainingTimesLastYear_4', 'TrainingTimesLastYear_5',
        'TrainingTimesLastYear_6', 'JobLevel', 'PerformanceRating'  # Added PerformanceRating
    ]

# The rest of your code remains the same until the form handling

# Create a form for user input
with st.form("prediction_form"):
    # Create two columns layout for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=70, value=30)
        BusinessTravel = st.selectbox("Business Travel", ["Rarely", "Frequently", "No Travel"])
        DailyRate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=500)
        Department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        DistanceFromHome = st.number_input("Distance From Home", min_value=1, max_value=30, value=5)
        Education = st.selectbox("Education", [1, 2, 3, 4, 5])
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        EnvironmentSatisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HourlyRate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
        JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
        JobLevel = st.number_input("Job Level", min_value=1, max_value=5, value=2)
        JobRole = st.selectbox("Job Role", ["Healthcare Representative", "Human Resources", "Laboratory Technician", 
                                        "Manager", "Manufacturing Director", "Research Director", 
                                        "Research Scientist", "Sales Executive", "Sales Representative"])
        JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 2)
    
    with col2:
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=4000)
        NumCompaniesWorked = st.number_input("Number of Companies Worked in", min_value=0, max_value=10, value=2)
        OverTime = st.selectbox("Over Time", ["Yes", "No"])
        PerformanceRating = st.slider("Performance Rating", 1, 4, 3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.number_input("Total Working Years", min_value=0, max_value=40, value=6)
        TrainingTimesLastYear = st.selectbox("Training Times Last Year", [0, 1, 2, 3, 4, 5, 6])
        WorkLifeBalance = st.slider("Work Life Balance", 1, 4, 2)
        YearsAtCompany = st.number_input("Years At Company", min_value=0, max_value=40, value=3)
        YearsInCurrentRole = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        YearsWithCurrManager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)

    # Submit button
    submit_button = st.form_submit_button("Predict Attrition")

# Make prediction when form is submitted
if submit_button:
    # Create a dictionary of form inputs
    input_dict = {
        'Age': int(Age),
        'BusinessTravel': str(BusinessTravel),
        'DailyRate': int(DailyRate),
        'Department': Department,
        'DistanceFromHome': int(DistanceFromHome),
        'Education': Education,
        'EducationField': str(EducationField),
        'EnvironmentSatisfaction': int(EnvironmentSatisfaction),
        'Gender': str(Gender),
        'HourlyRate': int(HourlyRate),
        'JobInvolvement': int(JobInvolvement),
        'JobLevel': int(JobLevel),
        'JobRole': JobRole,
        'JobSatisfaction': int(JobSatisfaction),
        'MaritalStatus': str(MaritalStatus),
        'MonthlyIncome': int(MonthlyIncome),
        'NumCompaniesWorked': int(NumCompaniesWorked),
        'OverTime': str(OverTime),
        'PerformanceRating': int(PerformanceRating),
        'RelationshipSatisfaction': int(RelationshipSatisfaction),
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': int(TotalWorkingYears),
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': int(WorkLifeBalance),
        'YearsAtCompany': int(YearsAtCompany),
        'YearsInCurrentRole': int(YearsInCurrentRole),
        'YearsSinceLastPromotion': int(YearsSinceLastPromotion),
        'YearsWithCurrManager': int(YearsWithCurrManager)
    }
    
    # Create DataFrame from input
    df = pd.DataFrame([input_dict])
    
    # Feature engineering - same as in Flask app
    df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                df['JobInvolvement'] +
                                df['JobSatisfaction'] +
                                df['RelationshipSatisfaction'] +
                                df['WorkLifeBalance']) / 5

    # Drop Columns
    df.drop(
        ['EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance'],
        axis=1, inplace=True)

    # Convert Total satisfaction into boolean
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply(lambda x: 1 if x >= 2.8 else 0)
    df.drop('Total_Satisfaction', axis=1, inplace=True)

    # Age boolean
    df['Age_bool'] = df['Age'].apply(lambda x: 1 if x < 35 else 0)
    df.drop('Age', axis=1, inplace=True)

    # Daily Rate boolean
    df['DailyRate_bool'] = df['DailyRate'].apply(lambda x: 1 if x < 800 else 0)
    df.drop('DailyRate', axis=1, inplace=True)

    # Department boolean
    df['Department_bool'] = df['Department'].apply(lambda x: 1 if x == 'Research & Development' else 0)
    df.drop('Department', axis=1, inplace=True)

    # Distance From Home boolean
    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1 if x > 10 else 0)
    df.drop('DistanceFromHome', axis=1, inplace=True)

    # Job Role boolean
    df['JobRole_bool'] = df['JobRole'].apply(lambda x: 1 if x == 'Laboratory Technician' else 0)
    df.drop('JobRole', axis=1, inplace=True)

    # Hourly Rate boolean
    df['HourlyRate_bool'] = df['HourlyRate'].apply(lambda x: 1 if x < 65 else 0)
    df.drop('HourlyRate', axis=1, inplace=True)

    # Monthly Income boolean
    df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply(lambda x: 1 if x < 4000 else 0)
    df.drop('MonthlyIncome', axis=1, inplace=True)

    # Number of Companies Worked boolean
    df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply(lambda x: 1 if x > 3 else 0)
    df.drop('NumCompaniesWorked', axis=1, inplace=True)

    # Total Working Years boolean
    df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply(lambda x: 1 if x < 8 else 0)
    df.drop('TotalWorkingYears', axis=1, inplace=True)

    # Years at Company boolean
    df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsAtCompany', axis=1, inplace=True)

    # Years in Current Role boolean
    df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply(lambda x: 1 if x < 3 else 0)
    df.drop('YearsInCurrentRole', axis=1, inplace=True)

    # Years Since Last Promotion boolean
    df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsSinceLastPromotion', axis=1, inplace=True)

    # Years With Current Manager boolean
    df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply(lambda x: 1 if x < 1 else 0)
    df.drop('YearsWithCurrManager', axis=1, inplace=True)

    # Convert Categorical to Numerical
    # Business Travel
    df['BusinessTravel_Rarely'] = 0
    df['BusinessTravel_Frequently'] = 0
    df['BusinessTravel_No_Travel'] = 0
    
    if BusinessTravel == 'Rarely':
        df['BusinessTravel_Rarely'] = 1
    elif BusinessTravel == 'Frequently':
        df['BusinessTravel_Frequently'] = 1
    else:
        df['BusinessTravel_No_Travel'] = 1
    df.drop('BusinessTravel', axis=1, inplace=True)

    # Education
    df['Education_1'] = 0
    df['Education_2'] = 0
    df['Education_3'] = 0
    df['Education_4'] = 0
    df['Education_5'] = 0
    
    if Education == 1:
        df['Education_1'] = 1
    elif Education == 2:
        df['Education_2'] = 1
    elif Education == 3:
        df['Education_3'] = 1
    elif Education == 4:
        df['Education_4'] = 1
    else:
        df['Education_5'] = 1
    df.drop('Education', axis=1, inplace=True)

    # Education Field
    df['EducationField_Life_Sciences'] = 0
    df['EducationField_Medical'] = 0
    df['EducationField_Marketing'] = 0
    df['EducationField_Technical_Degree'] = 0
    df['Education_Human_Resources'] = 0
    df['Education_Other'] = 0
    
    if EducationField == 'Life Sciences':
        df['EducationField_Life_Sciences'] = 1
    elif EducationField == 'Medical':
        df['EducationField_Medical'] = 1
    elif EducationField == 'Marketing':
        df['EducationField_Marketing'] = 1
    elif EducationField == 'Technical Degree':
        df['EducationField_Technical_Degree'] = 1
    elif EducationField == 'Human Resources':
        df['Education_Human_Resources'] = 1
    else:
        df['Education_Other'] = 1
    df.drop('EducationField', axis=1, inplace=True)

    # Gender
    df['Gender_Male'] = 0
    df['Gender_Female'] = 0
    
    if Gender == 'Male':
        df['Gender_Male'] = 1
    else:
        df['Gender_Female'] = 1
    df.drop('Gender', axis=1, inplace=True)

    # Marital Status
    df['MaritalStatus_Married'] = 0
    df['MaritalStatus_Single'] = 0
    df['MaritalStatus_Divorced'] = 0
    
    if MaritalStatus == 'Married':
        df['MaritalStatus_Married'] = 1
    elif MaritalStatus == 'Single':
        df['MaritalStatus_Single'] = 1
    else:
        df['MaritalStatus_Divorced'] = 1
    df.drop('MaritalStatus', axis=1, inplace=True)

    # Overtime
    df['OverTime_Yes'] = 0
    df['OverTime_No'] = 0
    
    if OverTime == 'Yes':
        df['OverTime_Yes'] = 1
    else:
        df['OverTime_No'] = 1
    df.drop('OverTime', axis=1, inplace=True)

    # Stock Option Level
    df['StockOptionLevel_0'] = 0
    df['StockOptionLevel_1'] = 0
    df['StockOptionLevel_2'] = 0
    df['StockOptionLevel_3'] = 0
    
    if StockOptionLevel == 0:
        df['StockOptionLevel_0'] = 1
    elif StockOptionLevel == 1:
        df['StockOptionLevel_1'] = 1
    elif StockOptionLevel == 2:
        df['StockOptionLevel_2'] = 1
    else:
        df['StockOptionLevel_3'] = 1
    df.drop('StockOptionLevel', axis=1, inplace=True)

    # Training Time Last Year
    df['TrainingTimesLastYear_0'] = 0
    df['TrainingTimesLastYear_1'] = 0
    df['TrainingTimesLastYear_2'] = 0
    df['TrainingTimesLastYear_3'] = 0
    df['TrainingTimesLastYear_4'] = 0
    df['TrainingTimesLastYear_5'] = 0
    df['TrainingTimesLastYear_6'] = 0
    
    if TrainingTimesLastYear == 0:
        df['TrainingTimesLastYear_0'] = 1
    elif TrainingTimesLastYear == 1:
        df['TrainingTimesLastYear_1'] = 1
    elif TrainingTimesLastYear == 2:
        df['TrainingTimesLastYear_2'] = 1
    elif TrainingTimesLastYear == 3:
        df['TrainingTimesLastYear_3'] = 1
    elif TrainingTimesLastYear == 4:
        df['TrainingTimesLastYear_4'] = 1
    elif TrainingTimesLastYear == 5:
        df['TrainingTimesLastYear_5'] = 1
    else:
        df['TrainingTimesLastYear_6'] = 1
    df.drop('TrainingTimesLastYear', axis=1, inplace=True)
    
    # Keep PerformanceRating as a numeric feature (this was missing in original code)
    # No need to drop it since it should be a feature the model was trained with
    
    # Ensure all expected columns are present and in the right order
    expected_columns = get_expected_columns()
    
    # Create an empty DataFrame with expected columns
    final_df = pd.DataFrame(columns=expected_columns)
    
    # Fill in values from our processed DataFrame
    for col in df.columns:
        if col in expected_columns:
            final_df[col] = df[col]
    
    # Fill any missing columns with 0
    final_df = final_df.fillna(0)
    
    # Debug information
    st.write(f"Number of features in final DataFrame: {final_df.shape[1]}")
    
    # Make prediction
    try:
        prediction = model.predict(final_df)
        
        # Display prediction with better styling
        st.divider()
        if prediction == 0:
            st.success("### Employee Might Not Leave The Job")
            st.balloons()
        else:
            st.error("### Employee Might Leave The Job")
            
        # Display feature importance (optional)
        if st.checkbox("Show Feature Analysis"):
            st.write("#### Key Factors That Might Be Influencing This Prediction")
            
            factors = []
            if df['Age_bool'].values[0] == 1:
                factors.append("Age below 35")
            if df['DailyRate_bool'].values[0] == 1:
                factors.append("Daily rate below 800")
            if df['Department_bool'].values[0] == 1:
                factors.append("Working in Research & Development")
            if df['DistanceFromHome_bool'].values[0] == 1:
                factors.append("Distance from home greater than 10")
            if df['JobRole_bool'].values[0] == 1:
                factors.append("Position as Laboratory Technician")
            if df['HourlyRate_bool'].values[0] == 1:
                factors.append("Hourly rate below 65")
            if df['MonthlyIncome_bool'].values[0] == 1:
                factors.append("Monthly income below 4000")
            if df['NumCompaniesWorked_bool'].values[0] == 1:
                factors.append("Worked at more than 3 companies")
            if df['TotalWorkingYears_bool'].values[0] == 1:
                factors.append("Less than 8 years of total working experience")
            if df['YearsAtCompany_bool'].values[0] == 1:
                factors.append("Less than 3 years at the company")
            if df['YearsInCurrentRole_bool'].values[0] == 1:
                factors.append("Less than 3 years in current role")
            if df['YearsSinceLastPromotion_bool'].values[0] == 1:
                factors.append("Less than 1 year since last promotion")
            if df['YearsWithCurrManager_bool'].values[0] == 1:
                factors.append("Less than 1 year with current manager")
            if df['Total_Satisfaction_bool'].values[0] == 0:
                factors.append("Lower overall satisfaction")
            if "OverTime_Yes" in df and df['OverTime_Yes'].values[0] == 1:
                factors.append("Works overtime")
            
            for i, factor in enumerate(factors):
                st.write(f"{i+1}. {factor}")
                
            if not factors:
                st.write("No significant risk factors identified.")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Debug information:")
        st.write(f"Expected columns: {len(expected_columns)}")
        st.write(f"Columns in final DataFrame: {len(final_df.columns)}")
        st.write("Missing columns:", set(expected_columns) - set(final_df.columns))
        st.write("Extra columns:", set(final_df.columns) - set(expected_columns))
        # Print out the exact column names for troubleshooting
        st.write("All expected columns:", expected_columns)

# Add sidebar with app information
with st.sidebar:
    st.title("About This App")
    st.write("This application predicts employee attrition based on various factors related to the employee's work history, job satisfaction, and personal information.")
    st.write("The model analyzes these inputs to determine if an employee might leave their job.")
    
    st.divider()
    st.subheader("How to use:")
    st.write("1. Enter the employee information in the form")
    st.write("2. Click 'Predict Attrition' to see the result")
    st.write("3. Check 'Show Feature Analysis' to see key factors")
    
    st.divider()
    st.write("Built with Streamlit")