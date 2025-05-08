"""
GitHub Issue Tagger - Streamlit UI Application
"""
import streamlit as st
import pandas as pd
import json
import os
import time
from issue_tagger import GitHubIssueTagging

# Page config
st.set_page_config(
    page_title="GitHub Issue Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS to improve UI appearance
st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .stProgress > div > div > div {
        background-color: #5C6BC0;
    }
    .highlight {
        padding: 0.3rem;
        background-color: #f0f2f6;
        border-radius: 0.3rem;
    }
    .tag-bug {
        color: white;
        background-color: #d73a4a;
        padding: 0.2em 0.6em;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .tag-feature {
        color: white;
        background-color: #0075ca;
        padding: 0.2em 0.6em;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .tag-question {
        color: white;
        background-color: #d876e3;
        padding: 0.2em 0.6em;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def validate_repo_url(url):
    """Validate GitHub repository URL and extract owner/repo"""
    import re
    pattern = r"github\.com\/([^\/]+)\/([^\/]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def format_tag(tag, confidence):
    """Format the tag with HTML styling"""
    tag_class = f"tag-{tag}"
    return f'<span class="{tag_class}">{tag.upper()}</span> <small>({confidence:.2f})</small>'

# Initialize tagger
@st.cache_resource
def get_tagger():
    return GitHubIssueTagging()

tagger = get_tagger()

# Title
st.title("🏷️ GitHub Issue Tagger")
st.markdown("""
This tool helps you automatically categorize GitHub issues as **bugs**, **features**, or **questions** 
using machine learning. Simply enter a GitHub repository URL below to get started.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This project uses the HuggingFace Transformers library to classify GitHub issues
    based on their content. It leverages a zero-shot classification model to determine
    if an issue is a bug report, feature request, or question.
    
    **Technologies used:**
    - Python
    - HuggingFace Transformers
    - GitHub API
    - Streamlit
    """)
    
    st.header("GitHub API Token (Optional)")
    github_token = st.text_input("Token", type="password", 
                                help="Provide a GitHub token to increase API rate limits")
    if github_token:
        tagger.set_github_token(github_token)
    
    st.header("Sample Repositories")
    sample_repos = {
        "TensorFlow": "https://github.com/tensorflow/tensorflow",
        "pandas": "https://github.com/pandas-dev/pandas",
        "React": "https://github.com/facebook/react",
        "VS Code": "https://github.com/microsoft/vscode"
    }
    
    for name, url in sample_repos.items():
        if st.button(name):
            st.session_state.repo_url = url

# Main input
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

repo_url = st.text_input("GitHub Repository URL:", 
                        value=st.session_state.repo_url,
                        placeholder="https://github.com/username/repository")

col1, col2 = st.columns(2)
with col1:
    max_issues = st.slider("Maximum number of issues to analyze:", 5, 100, 30)
with col2:
    show_existing_labels = st.checkbox("Show existing labels", value=True)

process_button = st.button("Analyze Issues")

# Function to load sample data
def load_sample_data():
    try:
        sample_path = os.path.join("sample_data", "sample_issues.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# Process repository
if process_button and repo_url:
    owner, repo = validate_repo_url(repo_url)
    
    if not owner or not repo:
        st.error("Invalid GitHub repository URL. Please enter a URL in the format: https://github.com/username/repository")
    else:
        with st.spinner(f"Fetching and analyzing issues from {owner}/{repo}..."):
            try:
                # Initialize progress bar
                progress_bar = st.progress(0)
                
                # Fetch issues
                issues = tagger.fetch_issues(owner, repo, max_issues=max_issues)
                
                if not issues:
                    st.warning("No issues found in this repository or reached API rate limit.")
                    # Try loading sample data
                    sample_data = load_sample_data()
                    if sample_data:
                        st.info("Using sample data instead.")
                        issues = sample_data
                    else:
                        st.stop()
                
                # Tag issues with progress bar
                results = []
                for i, issue in enumerate(issues):
                    # Update progress
                    progress = (i + 1) / len(issues)
                    progress_bar.progress(progress)
                    
                    # Process issue
                    issue_text = tagger.prepare_issue_text(issue)
                    category, confidence = tagger.classify_issue(issue_text)
                    
                    results.append({
                        "issue_number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "created_at": issue["created_at"].split("T")[0] if "created_at" in issue else "N/A",
                        "user": issue["user"]["login"] if "user" in issue and "login" in issue["user"] else "N/A",
                        "predicted_tag": category,
                        "confidence": confidence,
                        "has_labels": len(issue["labels"]) > 0 if "labels" in issue else False,
                        "existing_labels": [label["name"] for label in issue["labels"]] if "labels" in issue else []
                    })
                    
                    # Small delay to make progress bar visible
                    time.sleep(0.05)
                
                # Create DataFrame
                results_df = pd.DataFrame(results)
                
                # Complete progress
                progress_bar.progress(1.0)
                time.sleep(0.5)
                progress_bar.empty()
                
                # Display results
                st.subheader(f"Analysis Results for {owner}/{repo}")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    bug_count = sum(results_df["predicted_tag"] == "bug")
                    st.metric("Bugs", bug_count, f"{bug_count/len(results_df):.1%}")
                
                with col2:
                    feature_count = sum(results_df["predicted_tag"] == "feature")
                    st.metric("Features", feature_count, f"{feature_count/len(results_df):.1%}")
                
                with col3:
                    question_count = sum(results_df["predicted_tag"] == "question")
                    st.metric("Questions", question_count, f"{question_count/len(results_df):.1%}")
                
                # Display table
                st.markdown("### Tagged Issues")
                
                # Apply formatting to the table
                def format_row(row):
                    tag_html = format_tag(row["predicted_tag"], row["confidence"])
                    existing_labels = ""
                    if show_existing_labels and row["has_labels"]:
                        existing_labels = f"<br><small>Existing: {', '.join(row['existing_labels'])}</small>"
                    
                    return f"""
                    <tr>
                        <td>#{row['issue_number']}</td>
                        <td><a href="{row['url']}" target="_blank">{row['title']}</a>{existing_labels}</td>
                        <td>{row['user']}</td>
                        <td>{row['created_at']}</td>
                        <td>{tag_html}</td>
                    </tr>
                    """
                
                table_html = f"""
                <table class="dataframe">
                    <thead>
                        <tr>
                            <th>Issue #</th>
                            <th>Title</th>
                            <th>Author</th>
                            <th>Created</th>
                            <th>Predicted Tag</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(results_df.apply(format_row, axis=1))}
                    </tbody>
                </table>
                """
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Export options
                st.subheader("Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"{owner}_{repo}_tagged_issues.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_str = results_df.to_json(orient="records")
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name=f"{owner}_{repo}_tagged_issues.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Error processing repository: {e}")
elif process_button:
    st.warning("Please enter a GitHub repository URL.")
