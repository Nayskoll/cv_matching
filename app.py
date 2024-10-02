import streamlit as st
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import re

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)


# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Function to use GPT for analyzing CV aspects like soft skills and education level (no spelling score)
def analyze_cv_with_gpt(cv_text, job_description):
    prompt = f"""
    The following is a candidate's CV text:
    {cv_text}

    The following is the job description text:
    {job_description}

    Analyze the CV based on the following criteria:
    1. List the soft skills found in the CV, particularly those relevant to startups (autonomy, problem-solving, teamwork, leadership, communication).
    2. Evaluate the education level in the CV and compare it to what is required in the job description, placing particular emphasis on prestigious institutions and degrees in relevant fields.

    Provide a score out of 20, (10 is average, 5 is poor, 15 is good, 18 is excellent, 19 is exceptionnal) giving higher weight to candidates with strong education backgrounds and the most relevant soft skills for a startup environment. Write the score following this template : 'Overall score: XX/20' 
    """


    # Call OpenAI API using the new client format
    completion = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0.7,
    )

    # Correct access to the completion's text
    analysis = completion.choices[0].text.strip()

    # Use regex to extract the overall score from the GPT response
    score_match = re.search(r'(\d{1,2})\s*/\s*20', analysis)
    if score_match:
        overall_score = int(score_match.group(1))  # Extract the score as an integer
    else:
        overall_score = 0  # Default to 0 if no score is found

    return overall_score, analysis


# Function to combine embedding and GPT scores
def calculate_weighted_score(embedding_score, gpt_analysis_score, embedding_weight=0.7, gpt_weight=0.3):
    weighted_score = (embedding_score * embedding_weight) + ((gpt_analysis_score / 20) * gpt_weight)
    return weighted_score

# Function to process multiple CVs and match with job description
def generate_match_scores(cv_files, job_description, openai_api_key):
    # Select embeddings for job description
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    job_emb = embeddings.embed_documents([job_description])
    
    match_scores = []
    
    # Check if job description embeddings are valid
    if not job_emb or np.isnan(np.sum(job_emb)):
        st.error("Job description embedding is invalid. Please check the input.")
        return []

    # Process each CV
    for cv_file in cv_files:
        if cv_file is not None:
            if cv_file.name.endswith('.pdf'):
                cv_text = extract_text_from_pdf(cv_file)
            else:
                cv_text = cv_file.read().decode()

            # Skip if CV text is empty
            if not cv_text.strip():
                st.warning(f"CV {cv_file.name} contains no text or could not be read.")
                continue

            # Split CV into chunks (if necessary) and create embeddings
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            cv_chunks = text_splitter.create_documents([cv_text])
            cv_emb = embeddings.embed_documents([chunk.page_content for chunk in cv_chunks])

            # Skip if CV embeddings are invalid
            if not cv_emb or np.isnan(np.sum(cv_emb)):
                st.warning(f"Embedding for CV {cv_file.name} is invalid. Skipping.")
                continue

            # Calculate cosine similarity score between job and CV embeddings
            embedding_score = cosine_similarity(np.mean(job_emb, axis=0).reshape(1, -1), np.mean(cv_emb, axis=0).reshape(1, -1))[0][0]

            # Use GPT to analyze CV performance (soft skills and education)
            gpt_analysis_score, gpt_analysis = analyze_cv_with_gpt(cv_text, job_description)

            # Calculate final weighted score
            final_weighted_score = calculate_weighted_score(embedding_score, gpt_analysis_score)

            match_scores.append({
                "cv_name": cv_file.name,
                "embedding_score": embedding_score,
                "gpt_analysis_score": gpt_analysis_score,
                "final_weighted_score": final_weighted_score,
                "gpt_analysis": gpt_analysis
            })

    return match_scores

# File upload for job description (allow both PDF and TXT)
job_description_file = st.file_uploader('Upload a job description (PDF or TXT)', type=['pdf', 'txt'])

# Text input for supplementary information
query_text = st.text_area('Add supplementary information to the job description (optional):')

# File upload for multiple CVs
uploaded_cvs = st.file_uploader('Upload CVs (PDF or TXT)', type=['pdf', 'txt'], accept_multiple_files=True)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_cvs and (job_description_file or query_text)))
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_cvs and (job_description_file or query_text)))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating match scores...'):
            # Use job description from file if uploaded, otherwise use manual input
            if job_description_file:
                # Check if the uploaded file is a PDF or TXT and extract text accordingly
                if job_description_file.name.endswith('.pdf'):
                    job_description = extract_text_from_pdf(job_description_file)
                else:
                    job_description = job_description_file.read().decode()
            else:
                job_description = ""

            # Combine the job description file text with the additional information from the text area
            job_description += f"\n\n{query_text}"

            # Generate matching scores for CVs
            match_scores = generate_match_scores(uploaded_cvs, job_description, openai_api_key)
            result.extend(match_scores)
            del openai_api_key

# Display results
if len(result):
    st.write("Matching Scores and Analysis:")
    for res in result:
        st.write(f"**CV:** {res['cv_name']}")
        st.write(f"**Embedding Score:** {res['embedding_score']:.4f}")
        st.write(f"**Soft skills (GPT) Analysis Score (out of 20):** {res['gpt_analysis_score']:.2f}")
        st.write(f"**Final Weighted Score:** {res['final_weighted_score']:.4f}")
        st.write(f"**GPT Analysis:** {res['gpt_analysis']}")
        st.write("---")