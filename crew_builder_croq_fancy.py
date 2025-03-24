import os
import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
import pandas as pd
from datetime import datetime
import asyncio

# Add this near the top of the file, after the imports
if 'download_content' not in st.session_state:
    st.session_state.download_content = None

async def handle_crew_creation(agent_configs, human_input, groq_api_key):
    if not groq_api_key:
        st.error("Please enter your GROQ API key in the sidebar!")
        return

    try:
        with st.spinner("Creating and running your crew..."):
            os.environ["OPENAI_API_KEY"] = groq_api_key
            
            model = OpenAIChatCompletionsModel(
                model="llama-3.1-8b-instant",
                openai_client=AsyncOpenAI(base_url="https://api.groq.com/openai/v1")
            )

            tasklist, results = await create_and_run_crew(agent_configs, human_input, model)
            
            if tasklist is None:
                return

            # Display results in an organized way
            with results_container:
                st.markdown('<h3 class="section-header">Results</h3>', unsafe_allow_html=True)
                
                # Create tabs for different views
                results_tab1, results_tab2 = st.tabs(["Detailed Output", "Summary"])
                
                with results_tab1:
                    for i, result in enumerate(results):
                        with st.expander(f"Agent {i+1}: {agent_configs[i]['name']}", expanded=True):
                            st.markdown("**Task:**")
                            st.write(agent_configs[i]['instructions'])
                            st.markdown("**Output:**")
                            st.write(result.final_output)
                
                with results_tab2:
                    # Create a summary DataFrame
                    summary_data = []
                    for i, result in enumerate(results):
                        summary_data.append({
                            "Agent": f"{i+1}: {agent_configs[i]['name']}",
                            "Instructions": agent_configs[i]['instructions'],
                            "Output": result.final_output
                        })
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)

            # Generate the report content
            combined_text = f"""AUTONOMOUS CREW BUILDER - COMPLETE REPORT
{'='*50}

PART 1: CREW CONFIGURATION
{'='*50}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Additional Context:
{human_input}

Agent Configurations:
"""
            for i, config in enumerate(agent_configs):
                combined_text += f"\nAgent {i+1}: {config['name']}\n"
                combined_text += f"Instructions:\n{config['instructions']}\n"
                combined_text += "-" * 50 + "\n"

            combined_text += f"""
{'='*50}
PART 2: CREW RESULTS
{'='*50}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Results by Agent:
"""
            for i, result in enumerate(results):
                combined_text += f"\nAgent {i+1}: {agent_configs[i]['name']}\n"
                combined_text += f"Output:\n{result.final_output}\n"
                combined_text += "-" * 50 + "\n"
            
            # Store in session state
            st.session_state.download_content = combined_text
    except Exception as e:
        st.error(f"An error occurred while creating the crew: {str(e)}")
        st.error("Please check your GROQ API key and try again.")

async def create_and_run_crew(agent_configs, human_input, model):
    agentlist = []
    results = []
    
    for config in agent_configs:
        try:
            agent = Agent(
                name=config["name"],
                instructions=config["instructions"],
                model=model
            )
            agentlist.append(agent)
            
            # Run the agent with the task
            result = await Runner.run(agent, config["instructions"] + "\n\nAdditional Context: " + human_input)
            results.append(result)
            
        except Exception as e:
            st.error(f"Error creating agent: {str(e)}")
            return None, None

    return agentlist, results

# Set page config
st.set_page_config(
    page_title="Autonomous Crew Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .section-header {
        color: #1E3D59;
        padding: 1rem 0;
        border-bottom: 2px solid #1E3D59;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API key and general info
with st.sidebar:
    st.title("⚙️ Settings")
    groq_api_key = st.text_input('Enter your GROQ API key', type='password')
    st.markdown("---")
    st.markdown("""
        ### How to use this app:
        1. Enter your GROQ API key
        2. Provide the user input
        3. Go to the Configure Agents tab
        4. Define the number of agents
        5. Configure each agent's details
        6. Click 'Create Crew' to start
        7. Go to the Download tab to download configuration and results
    """)

# Main content
st.title('🤖 Autonomous Crew Builder')
st.markdown("""
    Create an autonomous crew of AI agents that work together to achieve your goals. 
    Each agent can be assigned specific roles, goals, and tasks.
""")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["User Input", "Configure Agents", "Download"])

# Create a container for the results (moved to the top)
results_container = st.container()

with tab1:
    st.markdown('<h3 class="section-header">User Input</h3>', unsafe_allow_html=True)
    human_input = st.text_area(
        "User Input",
        help="Enter any additional information or context that the agents should consider when executing their tasks",
        height=150
    )

with tab2:
    st.markdown('<h3 class="section-header">Agent Configuration</h3>', unsafe_allow_html=True)
    number_of_agents = st.number_input(
        'Number of Agents',
        min_value=1,
        max_value=10,
        value=1,
        help="Select how many agents you want in your crew"
    )

    # Create a container for agent configurations
    agent_configs = []
    for i in range(number_of_agents):
        with st.expander(f"Agent {i+1} Configuration", expanded=True):
            agent_name = st.text_input(f"Name", key=f"name_{i}")
            instructions = st.text_area(
                f"Instructions",
                key=f"instructions_{i}",
                help="Provide the complete instructions for this agent, including its role, goal, backstory, and expected output format.",
                height=200
            )
            
            agent_configs.append({
                "name": agent_name,
                "instructions": instructions
            })

    # Create Crew button (moved inside tab2)
    if st.button('🚀 Create Crew', type="primary"):
        asyncio.run(handle_crew_creation(agent_configs, human_input, groq_api_key))

# Download tab content (moved outside of results container)
with tab3:
    st.markdown('<h3 class="section-header">Download Report</h3>', unsafe_allow_html=True)
    if st.session_state.download_content is None:
        st.markdown("""
        After creating and running your crew, you can download the complete report here.
        
        The report will include:
        - Complete configuration of all agents
        - User input and context
        - Detailed results from each agent
        - Timestamps for configuration and execution
        
        Click 'Create Crew' in the Configure Agents tab to generate results, then return here to download them.
        """)
    else:
        st.markdown("""
        Your crew has been created and the results are ready for download.
        
        The report includes:
        - Complete configuration of all agents
        - User input and context
        - Detailed results from each agent
        - Timestamps for configuration and execution
        """)
        
        st.download_button(
            label="📥 Download Complete Report",
            data=st.session_state.download_content,
            file_name=f"crew_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        ) 