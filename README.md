# CV Matching and Evaluation with GPT and Langchain

This project is a web application that evaluates candidates' CVs by combining **embedding-based similarity** (using Langchain) and **GPT-based analysis** of soft skills and education. The app helps recruiters find the best candidates for startup environments by emphasizing both academic background and relevant soft skills.

The app is live and can be accessed at [cvmatching.streamlit.app](https://cvmatching.streamlit.app/).

## Features

- **CV Upload (PDF or TXT)**: Upload one or multiple CVs in PDF or TXT format for analysis.
- **Job Description Upload**: Upload a job description or enter supplementary details to enhance the evaluation.
- **Embedding-Based Score**: Uses Langchain embeddings to compare the CVs to the job description.
- **GPT-Based Analysis**: Leverages GPT to analyze the candidate's **soft skills** and **education background**, assigning a score out of 20, prioritizing strong candidates for startup environments.
- **Weighted Scoring**: Combines the embedding-based score and the GPT analysis score for a final result.

## How It Works

1. **Upload Job Description**: Upload a job description (PDF/TXT) or enter supplementary details manually.
2. **Upload CVs**: Upload multiple CVs (PDF/TXT) for evaluation.
3. **View Results**:
   - **Embedding-Based Score**: Measures how similar the CV is to the job description.
   - **GPT-Based Analysis**: Rates candidates on education and soft skills, giving a score out of 20.
   - **Final Weighted Score**: Combines both scores for a final evaluation.

## Live Demo

Try out the application live: [cvmatching.streamlit.app](https://cvmatching.streamlit.app/)

## Installation

### Prerequisites

- Python 3.8+
- A virtual environment (recommended)
- OpenAI API key

### Step-by-Step Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Nayskoll/cv_matching.git
    cd cv_matching
    ```

2. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    
    - Add your OpenAI API key:
    
      ```bash
      export OPENAI_API_KEY='your-openai-api-key-here'
      ```

5. **Run the app locally**:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in your browser at the URL provided by Streamlit (typically `http://localhost:8501`).
2. Upload your **job description** and **CVs**.
3. View the results with detailed analysis for each candidate.

### Results Breakdown

- **Embedding Score**: Measures the similarity between the job description and CV using cosine similarity.
- **GPT Analysis Score (out of 20)**: A strict evaluation of the candidate's soft skills and education level.
- **Final Weighted Score**: Combines both scores for a comprehensive evaluation.

## Example Output

- **CV:** `candidate1.pdf`
  - **Embedding Score:** 0.82
  - **GPT Analysis Score:** 18/20
  - **Final Weighted Score:** 0.68
  - **GPT Analysis:**
    - Soft skills: Autonomy, Problem-solving, Teamwork, Leadership
    - Education: Master's in Data Science from a prestigious university

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue on the [GitHub repository](https://github.com/Nayskoll/cv_matching.git).

---

Happy matching! 🎯