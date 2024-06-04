import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from getpass import getpass
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent 
from langchain.llms import OpenAI
import streamlit as st
import plotly.graph_objects as go
import scipy.stats as stats
from streamlit_option_menu import option_menu

#warnings.filterwarnings("ignore")

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide", page_title="EDA with LangChain and OpenAI GPT-3.5")

st.markdown("""
    <style>
    .main {background-color: #A1D6E2;}
    .stButton>button {background-color: #1E1E1E; color: white;}
    .stTextInput>div>div>input {background-color: #FBEAEB;}
    .sidebar {background-color: ##2F3C7E; }
    </style>
""", unsafe_allow_html=True)

st.title("Exploratory Data Analysis LLM")
st.title("with 🦜LangChain and OpenAI GPT-3.5")
st.markdown("#### Users can upload CSV files to the app, which then allows them to explore their datasets through natural language queries. By simply asking questions, users can obtain insights, generate visualizations, perform statistical analysis and test hypothesis without writing any code.")

openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

def extract_tool_inputs(result_dict):
    return [step[0].tool_input for step in result_dict['intermediate_steps'] if hasattr(step[0], 'tool_input')]

def run_commands(commands):
    for command in commands:
        try:
            if command.startswith("import"):
                exec(command, globals())
            else:
                result = eval(command, globals())
                if any(plot_type in command for plot_type in ["plot.bar", "'bar'", "sns.heatmap", "hist", "box", "kde", "pie", "scatter"]):
                    plt.figure(figsize=(10, 8))
                    exec(command, globals())
                    st.pyplot(plt.gcf())
                    plt.clf()
                elif isinstance(result, (list, dict)):
                    st.json(result)
                elif isinstance(result, plt.Figure):
                    st.pyplot(result)
                else:
                    exec(command)
                    st.write(result)
        except Exception as e:
            st.error(f"Error executing command {command}: {e}")

if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the dataframe
        st.write("Here are the first few rows of your dataset:")
        st.write(df.head())

        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, return_intermediate_steps=True, verbose=True)
        
        st.sidebar.header("EDA Options")
        option = st.sidebar.selectbox("Choose an action", ["Ask a question", "Show data", "Visualize data"])
        
        # Adding space in the sidebar
        st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        st.sidebar.markdown("##### by Udbhav Srivastava & Sreesh Reddy")

        if option == "Ask a question":
            st.header("Ask your questions about the dataset")
            user_input = st.text_input("Your question:")
            
            if user_input:
                with st.spinner('Processing...'):
                    result = agent(user_input)
                    commands = extract_tool_inputs(result)
                    run_commands(commands)
                    st.write(result['output'])

        elif option == "Show data":
            st.header("Dataset Overview")
            st.dataframe(df.describe())
            st.dataframe(df.info())

        elif option == "Visualize data":
            st.header("Visualize Data")
            plot_type = st.selectbox("Select plot type", ["Bar plot", "Heatmap", "Histogram", "Box plot", "KDE", "Pie chart", "Scatter plot"])
            column = st.selectbox("Select column", df.columns)

            if plot_type and column:
                if plot_type == "Bar plot":
                    fig = plt.figure(figsize=(10, 8))
                    df[column].value_counts().plot.bar()
                    st.pyplot(fig)
                elif plot_type == "Heatmap":
                    fig = plt.figure(figsize=(10, 8))
                    sns.heatmap(df.corr(), annot=True)
                    st.pyplot(fig)
                elif plot_type == "Histogram":
                    fig = plt.figure(figsize=(10, 8))
                    df[column].hist()
                    st.pyplot(fig)
                elif plot_type == "Box plot":
                    fig = plt.figure(figsize=(10, 8))
                    sns.boxplot(data=df[column])
                    st.pyplot(fig)
                elif plot_type == "KDE":
                    fig = plt.figure(figsize=(10, 8))
                    sns.kdeplot(df[column])
                    st.pyplot(fig)
                elif plot_type == "Pie chart":
                    fig = plt.figure(figsize=(10, 8))
                    df[column].value_counts().plot.pie()
                    st.pyplot(fig)
                elif plot_type == "Scatter plot":
                    column2 = st.selectbox("Select second column", df.columns)
                    if column2:
                        fig = plt.figure(figsize=(10, 8))
                        plt.scatter(df[column], df[column2])
                        st.pyplot(fig)
