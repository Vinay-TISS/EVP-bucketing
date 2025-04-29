# --- 0. Set Page First ---
import streamlit as st
st.set_page_config(page_title="EVP Bucketing Tool", layout="centered")

# --- 1. Import Required Libraries ---
import os
import zipfile
import gdown
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
import datetime
import torch
import random

# --- 2. Download Model from Google Drive if not exists ---
GDRIVE_FILE_ID = "1aWgld6R_psxnHZOOKInZRevplZUP8n4Z"  # ✅ Your correct Google Drive File ID

def download_model_from_drive():
    if not os.path.exists("local_model"):
        zip_path = "local_model.zip"
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("local_model")
        if os.path.exists(zip_path):
            os.remove(zip_path)

download_model_from_drive()

# --- 3. Load Transformer Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('./local_model/')

model = load_model()

# --- 4. Define Expanded EVP Pillars ---
pillars = {
    "Health & Wellbeing": "supporting physical, mental, emotional, and social health, happiness, positivity, mental peace, emotional strength, counseling support, healthy living, stress management, work-life harmony, mindfulness programs, physical wellness, health checkups, wellness benefits, medical insurance, therapy access, wellbeing assistance",
    
    "Financial Security & Benefits": "financial stability, income growth, salary increases, pay hikes, compensation fairness, bonuses, reward programs, stock options, provident fund, pension plans, insurance, financial safety nets, wealth building, savings support, income protection, retirement benefits, financial wellbeing",
    
    "Learning & Development": "continuous learning, personal development, career skill building, leadership programs, certifications, technical skills, professional growth, internal training, external workshops, self-improvement support, learning budgets, coaching and mentoring, cross-skilling, upskilling, future skills, knowledge sharing, peer learning, learning culture",
    
    "Career Growth & Opportunity": "career path clarity, promotion opportunities, leadership acceleration, job rotation programs, career mobility, internal hiring, succession planning, career counseling, leadership pipeline, merit-based promotions, visibility to management, growth mindset support, career ladders, career navigation help, fast-track promotions",
    
    "Flexibility & Work-Life Balance": "remote working, hybrid work options, work from home, flexible timings, compressed workweeks, sabbaticals, paid time off, autonomy over schedules, freedom to balance life, family-first policies, mental health days, freedom lifestyle support, supportive managers, asynchronous work culture",
    
    "Diversity, Equity & Inclusion (DEI)": "fair opportunities, inclusive hiring, respect for all backgrounds, belonging, celebrating diversity, no discrimination, gender equity, racial equity, accessible workplaces, LGBTQIA+ inclusion, neurodiversity inclusion, veteran inclusion, disability inclusion, culturally inclusive leadership, safe spaces, diverse leadership pipeline",
    
    "Work Culture & Psychological Safety": "open communication, honest feedback culture, transparent leadership, safe to voice opinions, non-toxic workplace, inclusive collaboration, team trust, approachable managers, respect in meetings, idea sharing without judgment, psychological comfort, mistakes as learning, emotional security at work, no politics workplace",
    
    "CSR & Purpose": "social impact projects, sustainability efforts, environmental volunteering, ethical business practices, giving back to society, employee volunteering programs, green initiatives, carbon neutrality goals, community engagement projects, social responsibility culture, impact-driven work, ethical labor practices",
    
    "Recognition & Rewards": "appreciation programs, performance bonuses, awards and honors, instant recognition, peer recognition, employee of the month awards, spot awards, celebration of milestones, visible leadership praise, manager thank-you notes, shout-outs, promotions based on merit, celebration of everyday wins",
    
    "People-First Identity": "human-centric leadership, treating employees with dignity, empathy in policies, respect for work-life boundaries, prioritizing people over profits, listening-first leadership style, humanizing the workplace, emotionally intelligent leadership, compassion during crises, personalized employee support",
    
    "Innovation & Entrepreneurship": "freedom to experiment, new idea encouragement, startup thinking, safe space for creativity, hackathons, innovation hubs, rapid prototyping culture, funding for employee ideas, intrapreneurship support, risk-taking encouragement, design thinking mindset, celebrating failed experiments",
    
    "Global Collaboration & Belonging": "cross-country teams, international mobility, work with diverse teams, global exposure, cross-border assignments, global family belonging, international project experiences, culturally adaptive leadership, language inclusion efforts, working towards a global purpose"
}

pillar_names = list(pillars.keys())
pillar_texts = list(pillars.values())
pillar_embeddings = model.encode(pillar_texts, convert_to_tensor=True)

# --- 5. Define Final Clean New Emerging EVP Themes ---
new_evp_theme_list = [
    "Future-Readiness and Agility",
    "Technology-Enabled Work Identity",
    "Dynamic Career Fluidity",
    "Self-Led Career Architecting",
    "Gig Mindset within Organizations",
    "Human-Tech Symbiosis",
    "Experiential and Adventure-First Work",
    "Multi-Identity Work Personas"
]

# --- 6. Streamlit User Interface ---
st.title("💡 EVP Bucketing Tool - Multi-Pillar Professional Version")
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
        emerging_indices = []

        # --- 7. Match Comments to Multiple Pillars ---
        for idx, comment in enumerate(comments):
            comment_embedding = model.encode(comment, convert_to_tensor=True)
            similarities = util.cos_sim(comment_embedding, pillar_embeddings)
            matched_pillars = []

            for i, similarity_score in enumerate(similarities[0]):
                if similarity_score.item() > 0.45:  # Threshold for multi-matching
                    matched_pillars.append(pillar_names[i])

            if matched_pillars:
                results.append((comment, matched_pillars))
            else:
                results.append((comment, "EMERGING"))
                emerging_texts.append(comment)
                emerging_indices.append(idx)

        # --- 8. BERTopic for Emerging Themes ---
        if emerging_texts:
            try:
                topic_model = BERTopic(embedding_model=model, verbose=False, low_memory=True, calculate_probabilities=False)
                topics, _ = topic_model.fit_transform(emerging_texts)

                topic_mapping = {}
                for idx, topic_num in enumerate(topics):
                    if idx < len(new_evp_theme_list):
                        mapped_theme = new_evp_theme_list[idx]
                    else:
                        mapped_theme = random.choice(new_evp_theme_list)
                    topic_mapping[idx] = f"NEW EVP: {mapped_theme}"

                # Update Results
                for idx, (comment, themes) in enumerate(results):
                    if themes == "EMERGING":
                        results[idx] = (comment, [topic_mapping.get(emerging_indices.pop(0), "NEW EVP: Future-Readiness and Agility")])

            except Exception as e:
                # If BERTopic crashes
                for idx, (comment, themes) in enumerate(results):
                    if themes == "EMERGING":
                        fallback_theme = random.choice(new_evp_theme_list)
                        results[idx] = (comment, [f"NEW EVP: {fallback_theme}"])

        # --- 9. Display Results ---
        st.write("### 🔍 Final EVP Theme Mapping Results")
        for comment, themes in results:
            if isinstance(themes, list):
                pillar_display = ", ".join(themes)
                st.markdown(f"**📝 {comment}** → _{pillar_display}_")
            else:
                st.markdown(f"**📝 {comment}** → _{themes}_")

        # --- 10. Save and Download Results ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"evp_bucketing_output_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for comment, themes in results:
                if isinstance(themes, list):
                    pillar_display = ", ".join(themes)
                    f.write(f"Comment: {comment}\nAssigned Themes: {pillar_display}\n{'-'*50}\n")
                else:
                    f.write(f"Comment: {comment}\nAssigned Theme: {themes}\n{'-'*50}\n")

        with open(filename, "rb") as f:
            st.download_button("📥 Download Result as TXT", f, file_name=filename, mime="text/plain")
