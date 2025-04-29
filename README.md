# EVP Bucketing Tool (NLP-based)

This is a Streamlit app that classifies employee or focus group comments into predefined EVP (Employee Value Proposition) pillars using semantic similarity. If the comment doesn't match any known theme, the app detects new "emerging" themes using BERTopic.

---

## 🎯 Features

- Classifies comments into 12 EVP pillars using Sentence-BERT
- Automatically detects emerging themes if no match found
- Results shown on-screen and downloadable as `.txt` file
- Easy to deploy on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🚀 To Run Locally

1. Install packages:
```bash
pip install -r requirements.txt
