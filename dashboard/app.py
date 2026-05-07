from __future__ import annotations

import os

import streamlit as st
import requests

API_URL = os.getenv("FINSIGHT_API_URL", "http://localhost:8000/api/v1")


def build_app() -> None:
    st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide")
    st.title("📈 FinSight AI — Financial Intelligence Assistant")

    with st.sidebar:
        st.header("Settings")
        route_override = st.selectbox("Force route (optional)", ["auto", "rag", "research", "memory"])
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("FinSight AI uses multi-agent RAG to answer financial queries from SEC filings, earnings calls, and market reports.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about earnings, filings, or market trends..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"query": prompt},
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]
                    route = data.get("route", "unknown")
                    st.markdown(answer)
                    st.caption(f"Routed via: `{route}`")
                except requests.RequestException as e:
                    answer = f"Error reaching FinSight API: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    build_app()


if __name__ == "__main__":
    main()
