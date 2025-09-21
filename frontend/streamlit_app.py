"""
Main Streamlit application for Resume Relevance Analysis System.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Relevance Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_score_color(score):
    """Get color class based on score."""
    if score >= 75:
        return "score-high"
    elif score >= 50:
        return "score-medium"
    else:
        return "score-low"


def format_score(score):
    """Format score with appropriate color."""
    color_class = get_score_color(score)
    return f'<span class="{color_class}">{score:.1f}%</span>'


# API helper functions
def call_api(endpoint, method="GET", data=None, files=None):
    """Make API calls with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code == 200 or response.status_code == 201:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server. Please ensure the backend is running."
    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Relevance Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Job Descriptions", "Resume Upload", "Analysis Center", "Results Viewer"]
    )
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Job Descriptions":
        show_job_descriptions()
    elif page == "Resume Upload":
        show_resume_upload()
    elif page == "Analysis Center":
        show_analysis_center()
    elif page == "Results Viewer":
        show_results_viewer()


def show_dashboard():
    """Display the main dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Fetch dashboard stats
    stats, error = call_api("/dashboard/stats")
    if error:
        st.error(error)
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", stats["total_resumes"])
    with col2:
        st.metric("Active Job Descriptions", stats["total_job_descriptions"])
    with col3:
        st.metric("Total Analyses", stats["total_analyses"])
    with col4:
        st.metric("Recent Analyses (7 days)", stats["recent_analyses"])
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Verdict Distribution")
        if stats["verdict_distribution"]:
            verdict_df = pd.DataFrame(
                list(stats["verdict_distribution"].items()),
                columns=["Verdict", "Count"]
            )
            fig = px.pie(verdict_df, values="Count", names="Verdict", 
                        title="Analysis Results Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No analysis data available yet.")
    
    with col2:
        st.subheader("Score Distribution")
        score_dist, error = call_api("/dashboard/score-distribution")
        if not error and score_dist:
            dist_data = score_dist["score_distribution"]
            score_df = pd.DataFrame(
                list(dist_data.items()),
                columns=["Score Range", "Count"]
            )
            fig = px.bar(score_df, x="Score Range", y="Count",
                        title="Score Range Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No score distribution data available.")
    
    # Recent analyses
    st.subheader("Recent Analyses")
    recent, error = call_api("/dashboard/recent-analyses?limit=10")
    if not error and recent:
        df = pd.DataFrame(recent)
        if not df.empty:
            df["final_score"] = df["final_score"].apply(lambda x: f"{x:.1f}%")
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                df[["candidate_name", "job_title", "company_name", "final_score", "verdict", "created_at"]],
                use_container_width=True
            )
        else:
            st.info("No recent analyses found.")
    else:
        st.info("No recent analyses available.")


def show_job_descriptions():
    """Display job descriptions management."""
    st.header("💼 Job Descriptions Management")
    
    tab1, tab2 = st.tabs(["View Job Descriptions", "Add New Job Description"])
    
    with tab1:
        st.subheader("Existing Job Descriptions")
        
        # Fetch job descriptions
        jds, error = call_api("/job-descriptions")
        if error:
            st.error(error)
            return
        
        if jds:
            for jd in jds:
                with st.expander(f"{jd['title']} - {jd.get('company_name', 'N/A')}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Company:** {jd.get('company_name', 'N/A')}")
                        st.write(f"**Location:** {jd.get('location', 'N/A')}")
                        st.write(f"**Experience Required:** {jd.get('experience_required', 'N/A')}")
                        st.write(f"**Created:** {jd['created_at'][:10]}")
                        
                        # Show truncated description
                        description = jd["description_text"]
                        if len(description) > 300:
                            description = description[:300] + "..."
                        st.write(f"**Description:** {description}")
                    
                    with col2:
                        st.metric("Analyses", jd.get("analysis_count", 0))
                        if jd.get("avg_score", 0) > 0:
                            st.metric("Avg Score", f"{jd['avg_score']:.1f}%")
                        
                        if st.button(f"View Analytics", key=f"analytics_{jd['id']}"):
                            show_job_analytics(jd['id'])
        else:
            st.info("No job descriptions found. Add one using the tab above.")
    
    with tab2:
        st.subheader("Add New Job Description")
        
        with st.form("add_job_description"):
            title = st.text_input("Job Title*", placeholder="e.g., Senior Software Engineer")
            company_name = st.text_input("Company Name", placeholder="e.g., TechCorp Inc.")
            location = st.text_input("Location", placeholder="e.g., New York, NY")
            experience_required = st.text_input("Experience Required", placeholder="e.g., 3-5 years")
            salary_range = st.text_input("Salary Range", placeholder="e.g., $80,000 - $120,000")
            description_text = st.text_area(
                "Job Description*",
                placeholder="Enter the complete job description...",
                height=300
            )
            created_by = st.text_input("Created By", placeholder="Your name")
            
            submit = st.form_submit_button("Add Job Description")
            
            if submit:
                if not title or not description_text:
                    st.error("Title and description are required.")
                else:
                    data = {
                        "title": title,
                        "company_name": company_name or None,
                        "location": location or None,
                        "experience_required": experience_required or None,
                        "salary_range": salary_range or None,
                        "description_text": description_text,
                        "created_by": created_by or None
                    }
                    
                    result, error = call_api("/job-descriptions", method="POST", data=data)
                    if error:
                        st.error(error)
                    else:
                        st.success("Job description added successfully!")
                        st.rerun()


def show_job_analytics(job_id):
    """Show analytics for a specific job."""
    analytics, error = call_api(f"/job-descriptions/{job_id}/analytics")
    if error:
        st.error(error)
        return
    
    st.subheader(f"Analytics for: {analytics['job_title']}")
    
    summary = analytics["summary"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Candidates", summary["total"])
    with col2:
        st.metric("Average Score", f"{summary['avg_score']:.1f}%")
    with col3:
        st.metric("High Suitability", summary["high"])
    with col4:
        st.metric("Medium Suitability", summary["medium"])
    
    if analytics["top_candidates"]:
        st.subheader("Top Candidates")
        candidates_df = pd.DataFrame(analytics["top_candidates"])
        candidates_df["final_score"] = candidates_df["final_score"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(candidates_df, use_container_width=True)


def show_resume_upload():
    """Display resume upload interface."""
    st.header("📁 Resume Upload")
    
    tab1, tab2 = st.tabs(["Upload New Resume", "Manage Resumes"])
    
    with tab1:
        st.subheader("Upload and Process Resume")
        
        with st.form("upload_resume"):
            uploaded_file = st.file_uploader(
                "Choose a resume file",
                type=["pdf", "docx"],
                help="Upload a PDF or DOCX file"
            )
            
            candidate_name = st.text_input("Candidate Name (optional)")
            candidate_email = st.text_input("Candidate Email (optional)")
            
            submit = st.form_submit_button("Upload Resume")
            
            if submit:
                if uploaded_file is None:
                    st.error("Please select a file to upload.")
                else:
                    with st.spinner("Uploading and processing resume..."):
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {}
                        if candidate_name:
                            data["candidate_name"] = candidate_name
                        if candidate_email:
                            data["candidate_email"] = candidate_email
                        
                        result, error = call_api("/resumes/upload", method="POST", files=files, data=data)
                        
                        if error:
                            st.error(error)
                        else:
                            st.success("Resume uploaded and processed successfully!")
                            st.json(result)
    
    with tab2:
        st.subheader("Uploaded Resumes")
        
        resumes, error = call_api("/resumes")
        if error:
            st.error(error)
            return
        
        if resumes:
            for resume in resumes:
                with st.expander(f"{resume.get('candidate_name', 'Unknown')} - {resume['file_name']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Email:** {resume.get('candidate_email', 'N/A')}")
                        st.write(f"**File Type:** {resume['file_type'].upper()}")
                        st.write(f"**Word Count:** {resume.get('word_count', 'N/A')}")
                        st.write(f"**Uploaded:** {resume['uploaded_at'][:10]}")
                        st.write(f"**Processed:** {'Yes' if resume['is_processed'] else 'No'}")
                    
                    with col2:
                        if st.button(f"View Details", key=f"details_{resume['id']}"):
                            show_resume_details(resume['id'])
                        
                        if st.button(f"Download", key=f"download_{resume['id']}"):
                            # Implementation for download would go here
                            st.info("Download functionality not implemented in this demo")
        else:
            st.info("No resumes uploaded yet.")


def show_resume_details(resume_id):
    """Show detailed resume information."""
    details, error = call_api(f"/resumes/{resume_id}")
    if error:
        st.error(error)
        return
    
    st.subheader(f"Resume Details: {details.get('candidate_name', 'Unknown')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information:**")
        st.write(f"- Name: {details.get('candidate_name', 'N/A')}")
        st.write(f"- Email: {details.get('candidate_email', 'N/A')}")
        st.write(f"- File: {details['file_name']}")
        st.write(f"- Type: {details['file_type'].upper()}")
        st.write(f"- Size: {details.get('word_count', 'N/A')} words")
    
    with col2:
        if details.get('contact_info'):
            st.write("**Contact Information:**")
            contact = details['contact_info']
            for key, value in contact.items():
                st.write(f"- {key.title()}: {value}")
    
    # Show text preview
    text_preview, error = call_api(f"/resumes/{resume_id}/text?text_type=cleaned")
    if not error and text_preview:
        st.subheader("Text Preview")
        preview_text = text_preview["text"]
        if len(preview_text) > 1000:
            preview_text = preview_text[:1000] + "..."
        st.text_area("Resume Content (Preview)", preview_text, height=200, disabled=True)


def show_analysis_center():
    """Display analysis creation interface."""
    st.header("🔍 Analysis Center")
    
    tab1, tab2 = st.tabs(["Single Analysis", "Bulk Analysis"])
    
    with tab1:
        st.subheader("Create Single Analysis")
        
        # Fetch resumes and job descriptions
        resumes, error1 = call_api("/resumes")
        jds, error2 = call_api("/job-descriptions")
        
        if error1 or error2:
            st.error("Error loading data for analysis.")
            return
        
        if not resumes or not jds:
            st.warning("You need at least one resume and one job description to perform analysis.")
            return
        
        with st.form("single_analysis"):
            # Resume selection
            resume_options = {f"{r.get('candidate_name', 'Unknown')} - {r['file_name']}": r['id'] for r in resumes}
            selected_resume = st.selectbox("Select Resume", list(resume_options.keys()))
            
            # Job description selection
            jd_options = {f"{jd['title']} - {jd.get('company_name', 'N/A')}": jd['id'] for jd in jds}
            selected_jd = st.selectbox("Select Job Description", list(jd_options.keys()))
            
            # Advanced configuration
            with st.expander("Advanced Configuration (Optional)"):
                hard_weight = st.slider("Hard Match Weight", 0.0, 1.0, 0.4, 0.1)
                semantic_weight = 1.0 - hard_weight
                st.write(f"Semantic Match Weight: {semantic_weight:.1f}")
                
                high_threshold = st.slider("High Score Threshold", 50, 100, 75)
                medium_threshold = st.slider("Medium Score Threshold", 0, high_threshold, 50)
            
            submit = st.form_submit_button("Start Analysis")
            
            if submit:
                resume_id = resume_options[selected_resume]
                jd_id = jd_options[selected_jd]
                
                config_overrides = {
                    "hard_match_weight": hard_weight,
                    "semantic_match_weight": semantic_weight,
                    "high_threshold": high_threshold,
                    "medium_threshold": medium_threshold
                }
                
                data = {
                    "resume_id": resume_id,
                    "job_description_id": jd_id,
                    "config_overrides": config_overrides
                }
                
                with st.spinner("Performing analysis... This may take a minute."):
                    result, error = call_api("/analyses", method="POST", data=data)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success("Analysis started successfully!")
                        st.info("The analysis is being processed in the background. Check the Results Viewer to see the completed analysis.")
                        st.json(result)
    
    with tab2:
        st.subheader("Bulk Analysis")
        st.info("Bulk analysis feature allows you to analyze multiple resumes against one job description.")
        
        # This would be similar to single analysis but with multiple resume selection
        # Implementation details omitted for brevity
        st.write("Feature coming soon...")


def show_results_viewer():
    """Display analysis results."""
    st.header("📊 Analysis Results")
    
    # Filters
    with st.expander("Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            verdict_filter = st.selectbox("Verdict", ["All", "High", "Medium", "Low"])
        with col2:
            min_score = st.number_input("Minimum Score", 0, 100, 0)
        with col3:
            max_score = st.number_input("Maximum Score", 0, 100, 100)
    
    # Construct filter parameters
    filter_params = []
    if verdict_filter != "All":
        filter_params.append(f"verdict={verdict_filter}")
    if min_score > 0:
        filter_params.append(f"min_score={min_score}")
    if max_score < 100:
        filter_params.append(f"max_score={max_score}")
    
    filter_string = "&" + "&".join(filter_params) if filter_params else ""
    
    # Fetch analyses
    analyses, error = call_api(f"/analyses?limit=50{filter_string}")
    if error:
        st.error(error)
        return
    
    if not analyses:
        st.info("No analyses found with the current filters.")
        return
    
    # Display results
    for analysis in analyses:
        with st.expander(f"{analysis['candidate_name']} → {analysis['job_title']} | Score: {analysis['final_score']:.1f}% | {analysis['verdict']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Candidate:** {analysis['candidate_name']}")
                st.write(f"**Job:** {analysis['job_title']} at {analysis.get('company_name', 'N/A')}")
                st.write(f"**Analysis Date:** {analysis['created_at'][:10]}")
                
                # Score visualization
                score = analysis['final_score']
                st.markdown(f"**Final Score:** {format_score(score)}", unsafe_allow_html=True)
                
                # Progress bar
                st.progress(score / 100)
            
            with col2:
                st.metric("Verdict", analysis['verdict'])
                
                if st.button(f"View Details", key=f"view_{analysis['id']}"):
                    show_analysis_details(analysis['id'])


def show_analysis_details(analysis_id):
    """Show detailed analysis results."""
    details, error = call_api(f"/analyses/{analysis_id}")
    if error:
        st.error(error)
        return
    
    st.subheader("📋 Detailed Analysis Results")
    
    # Score breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Score", f"{details['final_score']:.1f}%")
    with col2:
        st.metric("Hard Match", f"{details['hard_match_score']:.1f}%")
    with col3:
        st.metric("Semantic Match", f"{details['semantic_match_score']:.1f}%")
    
    # Verdict and confidence
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Verdict", details['verdict'])
    with col2:
        st.metric("Confidence", details.get('confidence_level', 'N/A'))
    
    # Feedback
    if details.get('feedback'):
        feedback = details['feedback']
        
        st.subheader("Executive Summary")
        st.write(feedback.get('executive_summary', 'No summary available'))
        
        st.subheader("Detailed Feedback")
        st.write(feedback.get('detailed_feedback', 'No detailed feedback available'))
        
        # Strengths and improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strengths")
            strengths = feedback.get('strengths_highlighted', [])
            if strengths:
                for strength in strengths:
                    st.write(f"✅ {strength}")
            else:
                st.write("No specific strengths identified")
        
        with col2:
            st.subheader("Areas for Improvement")
            improvements = feedback.get('areas_for_improvement', [])
            if improvements:
                for improvement in improvements:
                    st.write(f"📈 {improvement}")
            else:
                st.write("No specific improvements identified")
        
        # Recommendations
        st.subheader("Actionable Recommendations")
        recommendations = feedback.get('actionable_recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write("No specific recommendations available")


if __name__ == "__main__":
    main()