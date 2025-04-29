import streamlit as st
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
import datetime
import torch

# 1. Load Transformer model (small version, safe for GitHub/Streamlit Cloud)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')  # ✅ Smaller, faster, no upload issues

model = load_model()

# 2. Define EVP Pillars
pillars = {
    "Health & Wellbeing": "supporting physical, mental, emotional, and social health",
    "Financial Security & Benefits": "financial stability, savings, compensation, and insurance",
    "Learning & Development": "skill building, certifications, training, education programs",
    "Career Growth & Opportunity": "career pathways, internal mobility, leadership pipelines",
    "Flexibility & Work-Life Balance": "flexible working, hybrid models, personal autonomy",
    "Diversity, Equity & Inclusion (DEI)": "inclusion, diverse hiring, equitable opportunities",
    "Work Culture & Psychological Safety": "open communication, feedback, respectful culture",
    "CSR & Purpose": "social responsibility, sustainability, impact-driven work",
    "Recognition & Rewards": "employee rewards, celebrations, visible acknowledgment",
    "People-First Identity": "human-centric leadership, empathy, dignity for individuals",
    "Innovation & Entrepreneurship": "employee creativity, innovation, experimentation",
    "Global Collaboration & Belonging": "working across geographies, global teamwork, belonging"
}

pillar_names = list(pillars.keys())
pillar_texts = list(pillars.values())
pillar_embeddings = model.encode(pillar_texts, convert_to_tensor=True)

# 3. Streamlit UI
st.set_page_config(page_title="EVP Bucketing Tool", layout="centered")
st.title("💡 EVP Bucketing Tool")
st.markdown("**Step 1:** Paste employee comments below (one per line). Then press 'Generate EVP Themes'.")

with st.form("evp_form"):
    user_input = st.text_area("📥 Enter focus group / employee comments:", height=200)
    submitted = st.form_submit_button("🚀 Generate EVP Themes")

if submitted:
    if not user_input.strip():
        st.warning("Please enter at least one comment.")
    else:
        comments = [line.strip() for line in user_input.strip().split('\n') if line.strip()]
        results = []
        emerging_texts = []

        for comment in comments:
            comment_embedding = model.encode(comment, convert_to_tensor=True)
            similarities = util.cos_sim(comment_embedding, pillar_embeddings)
            best_pillar_idx = similarities.argmax().item()
            best_pillar = pillar_names[best_pillar_idx]
            confidence = similarities[0][best_pillar_idx].item()

            if confidence > 0.4:
                results.append((comment, best_pillar))
            else:
                results.append((comment, "EMERGING"))
                emerging_texts.append(comment)

       
# Run BERTopic only if enough data is available
if len(emerging_texts) >= 3:
    topic_model = BERTopic(verbose=False)
    topics, _ = topic_model.fit_transform(emerging_texts)

    for i, (comment, pillar) in enumerate(results):
        if pillar == "EMERGING":
            topic_index = topics.pop(0)
            topic_words = topic_model.get_topic(topic_index)
            if topic_words and isinstance(topic_words, list):
                new_theme = topic_words[0][0]  # Top word
                results[i] = (comment, f"EMERGING THEME: {new_theme}")
            else:
                results[i] = (comment, "UNKNOWN")
else:
    for i, (comment, pillar) in enumerate(results):
        if pillar == "EMERGING":
            results[i] = (comment, "UNKNOWN")  # Fallback label

        # Show Results
        st.write("### 🔍 EVP Theme Mapping Results")
        for comment, theme in results:
            st.markdown(f"**📝 {comment}** → _{theme}_")

        # Save to .txt
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"evp_bucketing_output_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for comment, theme in results:
                f.write(f"Comment: {comment}\nAssigned Theme: {theme}\n{'-'*50}\n")

        # Download button
        with open(filename, "rb") as f:
            st.download_button("📥 Download Result as TXT", f, file_name=filename, mime="text/plain")
