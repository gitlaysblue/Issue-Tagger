# GitHub Issue Tagger

A machine learning-powered tool that automatically categorizes GitHub issues as bugs, feature requests, or questions.

## Demo

![GitHub Issue Tagger Demo](https://via.placeholder.com/800x450.png?text=GitHub+Issue+Tagger+Demo)

## Features

- 🔍 Automatically analyze GitHub issues from any public repository
- 🏷️ Classify issues into three categories: bugs, feature requests, and questions
- 📊 View confidence scores for each classification
- 📋 Compare with existing labels (if any)
- 📊 Visual summary of issue distribution
- 💾 Export results as CSV or JSON

## How It Works

The GitHub Issue Tagger uses a pre-trained NLP model (Facebook's BART model with zero-shot classification) to analyze issue titles and descriptions. The model examines the text content and predicts the most likely category for each issue.

## Technologies Used

- Python 3.8+
- Hugging Face Transformers (for NLP model)
- GitHub API (for fetching issues)
- Streamlit (for UI)
- Pandas (for data handling)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/github-issue-tagger.git
   cd github-issue-tagger
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser at http://localhost:8501

3. Enter a GitHub repository URL (e.g., https://github.com/tensorflow/tensorflow)

4. Click "Analyze Issues" and view the results

## Optional: GitHub API Token

To avoid rate limiting, you can provide a GitHub API token in the sidebar. Create a token at https://github.com/settings/tokens with the `public_repo` scope.

## Project Structure

```
github-issue-tagger/
├── app.py               # Streamlit application
├── issue_tagger.py      # Core tagging functionality
├── requirements.txt     # Dependencies
├── README.md            # This file
└── sample_data/         # Sample data for offline testing
    └── sample_issues.json
```

## Future Improvements

- Add more classification categories
- Implement a fine-tuned model for better accuracy
- Add batch processing for large repositories
- Implement GitHub webhook integration for automatic tagging
- Add user authentication for accessing private repositories

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Made by : laysblue
