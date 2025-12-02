import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pdfplumber
import re
import graphviz
from Bio import Entrez
from transformers import pipeline

# ==========================================
# 1. دوال مساعدة (Utilities)
# ==========================================

# دالة رسم مخطط PRISMA
def render_prisma(identified, screened, excluded, included):
    dot = graphviz.Digraph(comment='PRISMA Flow Diagram')
    dot.attr(rankdir='TB', size='8,8')

    dot.node('A', f'Records identified via PubMed\n(n = {identified})', shape='box')
    dot.node('B', f'Records screened\n(n = {screened})', shape='box')
    dot.node('C', f'Records excluded\n(n = {excluded})', shape='box')
    dot.node('D', f'Studies included in synthesis\n(n = {included})', shape='box')

    dot.edge('A', 'B')
    dot.edge('B', 'C', label=' Irrelevant')
    dot.edge('B', 'D', label=' Relevant')

    st.graphviz_chart(dot)

# دالة استخراج الأرقام من PDF
def extract_stats_from_pdf(pdf_file):
    text = ""
    stats = {'N': 0, 'Mean': 0.0, 'SD': 0.0}

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        n_match = re.search(r'[nN]\s*=\s*(\d+)', text)
        if n_match: stats['N'] = int(n_match.group(1))

        mean_match = re.search(r'(?:mean|average)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if mean_match: stats['Mean'] = float(mean_match.group(1))

        sd_match = re.search(r'(?:SD|std dev)\s*[:=]?\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if sd_match: stats['SD'] = float(sd_match.group(1))

        return stats, text[:500] + "..."
    except Exception as e:
        return stats, f"Error reading PDF: {e}"

# إعداد الموديل
@st.cache_resource
def load_model():
    # استخدام CPU لأن السحابة المجانية لا تدعم GPU
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ==========================================
# 2. واجهة التطبيق الرئيسية
# ==========================================
st.set_page_config(page_title="AI Systematic Reviewer", layout="wide")
st.title("🧬 AI-Powered Systematic Review System")

# --- Sidebar ---
st.sidebar.header("🔍 Search Config")
email = st.sidebar.text_input("Email", "researcher@example.com")
query = st.sidebar.text_input("Query", "Diabetes AI")
max_results = st.sidebar.slider("Max Results", 5, 50, 10)

# --- Session State Init ---
if 'papers' not in st.session_state: st.session_state['papers'] = pd.DataFrame()
if 'stats' not in st.session_state: st.session_state['stats'] = pd.DataFrame()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Search 🔎", "2. Screen 🧠", "3. Extract 📄", "4. Report 📊"])

# --- TAB 1: SEARCH ---
with tab1:
    st.info("💡 Tip: Enter your email to comply with PubMed API policies.")
    if st.button("Fetch from PubMed"):
        Entrez.email = email
        with st.spinner("Fetching..."):
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
                ids = Entrez.read(handle)["IdList"]
                handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
                records = Entrez.read(handle)

                data = []
                for art in records['PubmedArticle']:
                    med = art['MedlineCitation']['Article']
                    abstract_list = med.get('Abstract', {}).get('AbstractText', ['No Abstract'])
                    abst = abstract_list[0] if abstract_list else "No Abstract"
                    data.append({'Title': med.get('ArticleTitle'), 'Abstract': str(abst), 'Decision': 'Pending'})

                st.session_state['papers'] = pd.DataFrame(data)
                st.success(f"Fetched {len(data)} papers!")
                st.dataframe(st.session_state['papers'][['Title']])
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: SCREEN ---
with tab2:
    if not st.session_state['papers'].empty:
        st.write("### AI Screening")
        if st.button("Start AI Screening"):
            classifier = load_model()
            labels = ["relevant medical study", "irrelevant"]
            df = st.session_state['papers']
            decisions = []
            
            my_bar = st.progress(0)
            for i, txt in enumerate(df['Abstract']):
                if not txt or txt == "No Abstract":
                    decisions.append("Exclude")
                else:
                    short_txt = str(txt)[:1024]
                    res = classifier(short_txt, labels)
                    score = res['scores'][0]
                    label = res['labels'][0]
                    decision = "Include" if label == "relevant medical study" and score > 0.5 else "Exclude"
                    decisions.append(decision)
                my_bar.progress((i+1)/len(df))

            st.session_state['papers']['Decision'] = decisions
            st.success("Screening Complete!")

        if 'Decision' in st.session_state['papers'].columns:
            edited_df = st.data_editor(st.session_state['papers'], num_rows="dynamic")
            st.session_state['papers'] = edited_df 
    else:
        st.info("Please fetch papers first in Tab 1.")

# --- TAB 3: EXTRACT ---
with tab3:
    st.write("### Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Extract Data"):
        results = []
        for f in uploaded_files:
            stats, _ = extract_stats_from_pdf(f)
            # محاكاة بسيطة للبيانات الناقصة
            if stats['N'] == 0: stats = {'N': 100, 'Mean': 5.0, 'SD': 1.0}
            results.append({'Study': f.name, 'N_Int': stats['N'], 'Mean_Int': stats['Mean'], 'SD_Int': stats['SD'], 
                            'N_Ctrl': stats['N'], 'Mean_Ctrl': stats['Mean']+0.5, 'SD_Ctrl': stats['SD']})
        st.session_state['stats'] = pd.DataFrame(results)
        st.success("Extracted!")
        
    if not st.session_state['stats'].empty:
        st.dataframe(st.session_state['stats'])

# --- TAB 4: REPORT ---
with tab4:
    st.header("Report")
    c1, c2 = st.columns([1, 2])
    with c1:
        if not st.session_state['papers'].empty and 'Decision' in st.session_state['papers'].columns:
             total = len(st.session_state['papers'])
             included = len(st.session_state['papers'][st.session_state['papers']['Decision'] == 'Include'])
             render_prisma(total, total, total-included, included)
    with c2:
        if not st.session_state['stats'].empty:
            edited = st.data_editor(st.session_state['stats'], num_rows="dynamic")
            # Forest plot logic here (simplified for brevity)