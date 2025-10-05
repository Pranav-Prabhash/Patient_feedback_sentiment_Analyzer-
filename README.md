🧠 Patient Feedback Sentiment Analyzer

A Streamlit web app that analyzes patient feedback and determines whether the sentiment is Positive or Negative using two models:

A Baseline Logistic Regression model (trained using TF-IDF)

A Fine-tuned DistilBERT model for advanced contextual sentiment understanding

This project is built for healthcare analytics and NLP learners who want to explore real-world sentiment analysis in healthcare domains.

🚀 Features

✅ Analyze a single feedback text instantly
✅ Upload a CSV of multiple patient reviews
✅ Get both Baseline (TF-IDF) and DistilBERT predictions side by side
✅ Download the analyzed output as a CSV file
✅ Lightweight Streamlit interface for easy interaction

🧩 Project Architecture
Patient-Feedback-Sentiment-Analyzer/
│
├── app.py                    # Streamlit main app
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignoring large model and cache files
│
├── data/
│   └── reviews.csv           # Sample test dataset
│ 
│
└── docs/
    └── project_documentation.docs   # Full project explanation 

📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Patient-Feedback-Sentiment-Analyzer.git
cd Patient-Feedback-Sentiment-Analyzer

2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

3️⃣ Install Required Packages
pip install -r requirements.txt

🧠 Model Setup

To keep this repository lightweight, the trained models are not included because of GitHub’s 100MB file limit.

You can download them separately below 👇

🔗 Model Links

DistilBERT Model (Fine-tuned on SST2) and Baseline TF-IDF Model (and vectorizer):
👉 Download from Hugging Face : https://huggingface.co/pranavp021/patient-feedback-sentiment-model/tree/main

After downloading, create a Models/ folder in your project directory and place the model files inside:

Patient-Feedback-Sentiment-Analyzer/
│
├── Models/
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_model.pkl
│   └── distilbert_sst2/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── vocab.txt
│       └── ...

▶️ Run the App Locally

After downloading the models and installing dependencies, run:

streamlit run app.py


Then open your browser at:

http://localhost:8501

📊 Sample Input (reviews.csv)

Example file structure for batch prediction:

text
The doctor was very friendly and helpful.
The hospital staff were rude and unprofessional.
Waiting time was too long but the nurse was caring.
The treatment was effective and affordable.
I had a terrible experience with the billing department.

💾 Output Format

When you upload the CSV, you can download the analyzed results as output.csv with the following columns:

text	Baseline	DistilBERT
The doctor was rude.	Negative	Negative
Staff was caring and kind.	Positive	Positive
🧠 Tech Stack
Component	Technology
Programming	Python 3.x
Web App	Streamlit
NLP	Hugging Face Transformers
Model	DistilBERT fine-tuned on SST2
ML Baseline	Logistic Regression with TF-IDF
Data Handling	Pandas, Numpy
Visualization	Matplotlib (optional)
🌐 Live Demo

(If you host on Hugging Face Spaces or Streamlit Cloud, add the link below)
👉 Try the Live App Here

🧾 License

This project is released under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.

👨‍💻 Author

Pranav P
📍 NLP & Data Science Enthusiast | Healthcare AI Explorer
pranavprabhash93@gmail.com
🔗 LinkedIn : https://www.linkedin.com/in/pranav021/

🔗 GitHub : https://github.com/Pranav-Prabhash
