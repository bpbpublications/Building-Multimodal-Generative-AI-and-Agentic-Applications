# frontend/app.py

import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run_query  # 👈 This now works when executed from project root

st.set_page_config(page_title="GenAI Bot", page_icon="🤖")

st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="color:white; background-color:#26b7c9; padding:15px; border-radius:10px; font-size:22px;">🤖 GenAI Bot</div>', unsafe_allow_html=True)

user_query = st.text_input("Ask something...", placeholder="Find all users who purchased shoes last month.")

if st.button("🚀 Submit Query") and user_query:
    with st.spinner("Thinking..."):
        result = run_query(user_query)

        st.markdown("### 📋 Final Aggregated Summary:")
        st.markdown(f"<div class='big-font'>{result['aggregated_result']['final_summary']}</div>", unsafe_allow_html=True)

        st.markdown("### 📑 Details:")
        for detail in result['aggregated_result']['details']:
            st.markdown(f"**Summary:** {detail['summary']}")
            st.markdown(f"**Data:**\n```{detail['original_data']}```")

        st.markdown("### 🗃️ Generated SQL Query:")
        st.code(result["sql_query"], language="sql")

        st.markdown("### ✅ SQL Query Grade:")
        st.markdown(result["sql_grade"])

        st.markdown("### ✅ Summary Grade:")
        st.markdown(result["summary_grade"])
