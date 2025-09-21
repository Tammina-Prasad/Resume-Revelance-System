"""
Main Streamlit app for Resume Relevance Analysis System
This is the entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'frontend'))
sys.path.append(os.path.join(current_dir, 'backend'))

# Import and run the main app
try:
    from frontend.streamlit_app import main
    main()
except ImportError as e:
    import streamlit as st
    st.error(f"Import error: {e}")
    st.info("This is a demo of the Resume Relevance Analysis System")
    st.write("The full application requires backend services to be running.")