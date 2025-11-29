# app.py

import streamlit as st
import pandas as pd

from resume_scoring import extract_text_from_file, score_resume_against_jd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Resume Screening Agent",
    layout="wide",
    page_icon="🧠",
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🧠 Resume Agent")
    st.markdown(
        """
This AI agent:

- Reads the **Job Description**
- Analyzes **multiple resumes**
- Computes a **0–100 fit score**
- Highlights **strengths & gaps**

**Steps to use:**
1. Paste the JD  
2. Upload PDFs  
3. Click **Rank Resumes**  
        """
    )
    st.markdown("---")
    st.markdown("Built with **Hugging Face embeddings** + **Streamlit**.")

# ---------- HEADER ----------
st.title("AI Resume Screening Agent")
st.caption("Rank candidates automatically based on semantic fit with your Job Description.")

st.markdown("")

# ---------- MAIN INPUT AREA ----------
col_jd, col_upload = st.columns([3, 2], gap="large")

with col_jd:
    st.subheader("1️⃣ Paste Job Description")
    jd_text = st.text_area(
        "Job Description",
        placeholder="Paste the full JD here (role, responsibilities, required skills, experience...)",
        height=260,
        label_visibility="collapsed",
    )

with col_upload:
    st.subheader("2️⃣ Upload Resumes (PDF / Text)")
    uploaded_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown(
        "<small>Tip: You can select multiple files at once.</small>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------- ACTION BUTTON ----------
center_col = st.columns([1, 1, 1])[1]
with center_col:
    run_clicked = st.button("⚙️ Rank Resumes", use_container_width=True)

# ---------- MAIN LOGIC ----------
if run_clicked:
    if not jd_text.strip():
        st.error("Please paste a Job Description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            candidate_name = uploaded_file.name

            status.info(f"Processing **{candidate_name}** ({idx}/{len(uploaded_files)})...")

            resume_text = extract_text_from_file(uploaded_file)

            if not resume_text.strip():
                st.warning(f"Could not extract text from **{candidate_name}**. Skipping.")
                continue

            scoring = score_resume_against_jd(jd_text, resume_text)

            results.append({
                "Candidate": candidate_name,
                "Score": scoring["score"],
                "Summary": scoring["summary"],
                "Strengths": scoring["strengths"],
                "Gaps": scoring["gaps"],
            })

            progress.progress(idx / len(uploaded_files))

        status.success("Processing complete ✅")

        if not results:
            st.warning("No results generated. Check if resumes were readable.")
        else:
            # ---------- SUMMARY METRICS ----------
            df = pd.DataFrame(results).sort_values(
                by="Score", ascending=False
            ).reset_index(drop=True)
            df.index = df.index + 1  # rank starting from 1

            st.subheader("3️⃣ Overview & Rankings")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Candidates", len(df))
            with c2:
                st.metric("Best Score", f"{df['Score'].max()} / 100")
            with c3:
                st.metric("Average Score", f"{int(df['Score'].mean()):d} / 100")

            st.markdown("#### Ranked Candidates")
            st.dataframe(
                df[["Score", "Candidate", "Summary"]],
                use_container_width=True,
                height=260,
            )

            # ---------- DETAILED VIEW ----------
            st.markdown("#### 4️⃣ Detailed Analysis per Candidate")

            def score_badge(score: int) -> str:
                if score >= 70:
                    color = "#16a34a"  # green
                elif score >= 50:
                    color = "#f59e0b"  # orange
                else:
                    color = "#dc2626"  # red
                return f"""
                <span style="
                    background-color:{color};
                    color:white;
                    padding:2px 8px;
                    border-radius:999px;
                    font-size:0.8rem;">
                    {score} / 100
                </span>
                """

            for rank, row in df.iterrows():
                with st.expander(f"#{rank} – {row['Candidate']}"):
                    st.markdown(
                        f"**Overall Score:** "
                        f"{score_badge(int(row['Score']))}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Summary:** {row['Summary']}")
                    st.markdown("**Strengths:**")
                    st.text(row["Strengths"])
                    st.markdown("**Gaps / Concerns:**")
                    st.text(row["Gaps"])

            # ---------- DOWNLOAD ----------
            st.markdown("---")
            csv = df.to_csv(index_label="Rank").encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )
