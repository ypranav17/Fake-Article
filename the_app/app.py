import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pipeline.full_pipeline import run_pipeline_from_url, run_pipeline_from_text_manual


st.set_page_config(page_title="Fake News Detector", layout="wide")

# === Custom Theme ===
st.markdown("""
    <style>
        .main { background-color: #111; }
        h1, h2, h3, h4, h5, h6, .stMarkdown { color: white; }
        small { color: #aaa; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Real-Time Fake News Detector")

# === Sidebar Settings ===
st.sidebar.header("⚙️ Settings")
min_score = st.sidebar.slider("Minimum Claim Score", 0.0, 1.0, 0.6, 0.05)
max_claims = st.sidebar.slider("Maximum Claims to Display", 5, 50, 10)
only_fake = st.sidebar.checkbox("Show only FAKE claims", value=False)

# === Input Mode ===
input_mode = st.radio("Select Input Mode:", ("Article URL", "Manual Text Input"))
if input_mode == "Article URL":
    url = st.text_input("📎 Article URL:", "")
    user_input = url
else:
    user_input = st.text_area("✍️ Enter Text Manually:", height=200)

# === Initialize State ===
if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False
if "results" not in st.session_state:
    st.session_state.results = []

# === Analyze Button ===
if st.button("Analyze") and user_input:
    with st.spinner("Running claim detection pipeline..."):
        start = time.time()
        if input_mode == "Article URL":
            st.session_state.results = run_pipeline_from_url(user_input)
        else:
            from pipeline.full_pipeline import run_pipeline_from_text_manual
            st.session_state.results = run_pipeline_from_text_manual(user_input)

        st.session_state.elapsed_time = time.time() - start
    st.session_state.analysis_requested = True


# === Run Analysis ===
if st.session_state.analysis_requested:
    results = [r for r in st.session_state.results if r["score"] >= min_score]
    if only_fake:
        results = [r for r in results if r["claim_type"] == "FAKE"]

    fake_count = sum(1 for r in results if r['claim_type'] == "FAKE")
    real_count = sum(1 for r in results if r['claim_type'] == "REAL")
    unsure_count = sum(1 for r in results if r['claim_type'] == "UNSURE")
    total = fake_count + real_count

    # ✅ Verdict Logic
    if total == 0:
        verdict = "UNSURE"
    elif fake_count > real_count:
        verdict = "FAKE"
    else:
        verdict = "REAL"

    color_map = {"FAKE": "red", "REAL": "green", "UNSURE": "gray"}
    verdict_color = color_map.get(verdict, "gray")
    percent = (fake_count / total * 100) if total else 0

    # === Tabbed Layout ===
    tabs = st.tabs(["📊 Overview", "🧾 Claims", "🔍 Evidence", "⬇️ Downloads"])

    with tabs[0]:
        st.subheader("Primary Verdict")
        st.markdown(f"<h1 style='color: {verdict_color}; font-size: 64px'>{verdict}</h1>", unsafe_allow_html=True)

        if verdict == "REAL":
            st.success(f"📘 Faker: {percent:.2f}%")
        elif verdict == "FAKE":
            st.error(f"📘 Faker: {percent:.2f}%")
        else:
            st.info("📘 Not enough data to make a decision.")

        st.markdown(f"⏱️ Processed in `{st.session_state.elapsed_time:.2f}` seconds")

        # 🧠 Explanation
        st.markdown("### 🧠 Why this verdict?")
        st.info(f"The article contains `{fake_count}` potentially fake claims out of `{total}`. "
                f"This indicates a {'high' if percent > 50 else 'moderate'} presence of misinformation.")

        # 🟢 Pie chart (centered)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            pie_labels = []
            pie_values = []
            pie_colors = []

            if fake_count > 0:
                pie_labels.append("FAKE")
                pie_values.append(fake_count)
                pie_colors.append("red")

            if real_count > 0:
                pie_labels.append("REAL")
                pie_values.append(real_count)
                pie_colors.append("green")

            if unsure_count > 0:
                pie_labels.append("UNSURE")
                pie_values.append(unsure_count)
                pie_colors.append("gray")

            if pie_values:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors)
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)
            else:
                st.warning("⚠️ No claims detected with the current filters. Adjust your score threshold or check your input.")

    with tabs[1]:
        st.subheader("Top Checked Claims")
        top_claims = sorted(results, key=lambda x: x["score"], reverse=True)[:max_claims]
        for r in top_claims:
            claim_color = color_map.get(r["claim_type"], "gray")
            st.markdown(
                f"<span style='color:{claim_color}'>●</span> "
                f"**{r['sentence']}**  \n"
                f"<small>Score: `{round(r['score'], 2)}` | Type: `{r['claim_type']}`</small>",
                unsafe_allow_html=True
            )

    with tabs[2]:
        st.subheader("🔍 Evidence Sources")
        for r in results[:max_claims]:
            if r.get("evidence") and len(r["evidence"]) > 0:
                st.markdown(f"**📝 Claim:** {r['sentence']}")
                st.markdown(f"- 🌐 [Source Link]({r['evidence'][0]})")

    with tabs[3]:
        st.subheader("⬇️ Download Results")
        export_data = [{"sentence": r["sentence"], "score": r["score"], "claim_type": r["claim_type"]} for r in results]
        json_data = json.dumps(export_data, indent=2)
        st.download_button("Download as JSON", json_data, file_name="results.json", mime="application/json")

        df = pd.DataFrame(export_data)
        csv = df.to_csv(index=False)
        st.download_button("Download as CSV", csv, file_name="results.csv", mime="text/csv")
