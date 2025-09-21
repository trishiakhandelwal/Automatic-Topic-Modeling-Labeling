import streamlit as st
import pandas as pd
import numpy as np
import io
from pipeline import run_pipeline, query_documents
import streamlit.components.v1 as components

st.set_page_config(page_title="Topic Modeling and Document Retrieval", layout="wide")

st.title("Topic Modeling and Document Retrieval")
st.write("Upload a CSV with a `content` column containing your files to cluster into topics and generate labels.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    file_buffer = io.StringIO(uploaded_file.getvalue().decode("utf-8"))

    # Only run the pipeline once per uploaded file
    if "pipeline_done" not in st.session_state or st.session_state["last_file_bytes"] != uploaded_file.getvalue():
        with st.spinner("Processing your file... this may take a while"):
            df, topic_docs_mapping, df_topic_names, embeddings = run_pipeline(file_buffer)
        st.session_state["df"] = df
        st.session_state["topic_docs_mapping"] = topic_docs_mapping
        st.session_state["df_topic_names"] = df_topic_names
        st.session_state["embeddings"] = embeddings
        st.session_state["pipeline_done"] = True
        st.session_state["last_file_bytes"] = uploaded_file.getvalue()

    df = st.session_state["df"]
    topic_docs_mapping = st.session_state["topic_docs_mapping"]
    df_topic_names = st.session_state["df_topic_names"]
    embeddings = st.session_state["embeddings"]

    query_text = st.text_input("Search documents (query):")
    if query_text:
        top_docs = query_documents(query_text, df, embeddings, top_n=7)
        st.header(f"Top documents for your query: '{query_text}'")

        num_per_row = 4
        for i in range(0, len(top_docs), num_per_row):
            cols = st.columns(num_per_row)
            for j, (_, doc_row) in enumerate(top_docs.iloc[i:i+num_per_row].iterrows()):
                with cols[j]:
                    st.markdown(
                        f"<div style='font-size:12px;line-height:1.4;'>{doc_row['content'][:200]}...</div>",
                        unsafe_allow_html=True
                    )

                    with st.expander("View Full"):
                        st.markdown(
                            f"<div style='font-size:12px;line-height:1.4;word-wrap: break-word;overflow-wrap: break-word;white-space: pre-wrap;word-break: break-word;max-width: 100%;'>{doc_row['full_content']}</div>",
                            unsafe_allow_html=True
                        )

    st.header("Topics")
    cols_per_row = 3
    for i in range(0, len(df_topic_names), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (_, row) in enumerate(df_topic_names.iloc[i:i+cols_per_row].iterrows()):
            with cols[j]:
                if st.button(row["Name"], key=f"topic_{row['Topic']}"):
                    st.session_state["selected_topic_id"] = row["Topic"]

    if "selected_topic_id" in st.session_state:
        topic_id = st.session_state["selected_topic_id"]
        topic_name = df_topic_names[df_topic_names["Topic"] == topic_id]["Name"].values[0]

        st.subheader(f"Summaries for topic: {topic_name}")
        topic_df = topic_docs_mapping[topic_id]

        num_per_row = 4
        for i in range(0, len(topic_df), num_per_row):
            cols = st.columns(num_per_row)
            for j, (_, doc_row) in enumerate(topic_df.iloc[i:i+num_per_row].iterrows()):
                with cols[j]:
                    st.markdown(
                        f"<div style='font-size: 12px; line-height: 1.4;'>{doc_row['content'][:200]}...</div>",
                        unsafe_allow_html=True
                    )
                    with st.expander("View Full"):
                        st.markdown(
                            f"<div style='font-size:12px;line-height:1.4;'>{doc_row['content']}</div>",
                            unsafe_allow_html=True
                        )