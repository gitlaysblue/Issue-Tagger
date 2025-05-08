"""
GitHub Issue Tagger - Core functionality for classifying GitHub issues
"""
import re
import requests
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Any, Tuple


class GitHubIssueTagging:
    """Class to handle GitHub issue tagging"""
    
    def __init__(self):
        """Initialize the tagger with a pre-trained model"""
        # Using a zero-shot classification pipeline as it doesn't require fine-tuning
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # Use CPU for inference
        )
        self.categories = ["bug", "feature", "question"]
        self.github_token = None  # Optional for increased API rate limits
    
    def set_github_token(self, token: str):
        """Set GitHub token for API access with higher rate limits"""
        self.github_token = token
        
    def fetch_issues(self, owner: str, repo: str, max_issues: int = 30) -> List[Dict[str, Any]]:
        """Fetch issues from a GitHub repository"""
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
            
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {"state": "open", "per_page": max_issues}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            issues = response.json()
            
            # Filter out pull requests which are also returned by the API
            issues = [issue for issue in issues if "pull_request" not in issue]
            
            return issues
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issues: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for the model"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove code blocks for cleaner text
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_issue_text(self, issue: Dict[str, Any]) -> str:
        """Combine issue title and body for better classification"""
        title = issue.get("title", "")
        body = issue.get("body", "")
        
        # Preprocess both title and body
        title_clean = self.preprocess_text(title)
        body_clean = self.preprocess_text(body)
        
        # Combine with more weight on the title
        combined_text = f"{title_clean} {title_clean} {body_clean}"
        return combined_text[:1024]  # Limit length for model input
    
    def classify_issue(self, issue_text: str) -> Tuple[str, float]:
        """Classify issue text into a category and return confidence"""
        if not issue_text.strip():
            return "question", 0.33  # Default for empty text
        
        result = self.classifier(issue_text, self.categories)
        
        # Get the highest scoring category and its score
        best_category = result["labels"][0]
        confidence = result["scores"][0]
        
        return best_category, confidence
    
    def tag_issues(self, issues: List[Dict[str, Any]]) -> pd.DataFrame:
        """Tag a list of issues and return as a DataFrame"""
        results = []
        
        for issue in issues:
            issue_text = self.prepare_issue_text(issue)
            category, confidence = self.classify_issue(issue_text)
            
            results.append({
                "issue_number": issue["number"],
                "title": issue["title"],
                "url": issue["html_url"],
                "created_at": issue["created_at"],
                "user": issue["user"]["login"],
                "predicted_tag": category,
                "confidence": confidence,
                "has_labels": len(issue["labels"]) > 0,
                "existing_labels": [label["name"] for label in issue["labels"]]
            })
        
        return pd.DataFrame(results)


# For testing
if __name__ == "__main__":
    tagger = GitHubIssueTagging()
    test_issues = tagger.fetch_issues("tensorflow", "tensorflow", max_issues=5)
    
    if test_issues:
        results_df = tagger.tag_issues(test_issues)
        print(results_df[["issue_number", "title", "predicted_tag", "confidence"]])
    else:
        print("No issues fetched. Check your internet connection or GitHub API rate limits.")
