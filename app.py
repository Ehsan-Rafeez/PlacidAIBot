
"""
PlacidWay Streamlit Chat Application.
"""

import streamlit as st

from src.chatbot import ask_chatbot
from src.version import APP_VERSION


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="PlacidWay AI Assistant",
    page_icon="🌍",
    layout="centered"
)

st.markdown(
    f"""
    <style>
        .app-version {{
            position: fixed;
            top: 3.5rem;
            right: 1rem;
            z-index: 999;
            padding: 0.25rem 0.6rem;
            border: 1px solid #718096;
            border-radius: 0.5rem;
            background: var(--background-color, #ffffff);
            color: var(--text-color, #263238);
            font-size: 0.8rem;
            pointer-events: none;
        }}
    </style>
    <span class="app-version" aria-label="Application version {APP_VERSION}">v{APP_VERSION}</span>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------
# Application header
# ---------------------------------------

st.title("🌍 PlacidWay AI Assistant")

st.caption(
    "Explore medical treatments, clinics, "
    "packages, and costs from approved "
    "PlacidWay website information."
)

st.info(
    "Answers are based on the seven approved "
    "PlacidWay pages. For medical decisions, "
    "consult a qualified healthcare professional."
)


# ---------------------------------------
# Initialize chat history
# ---------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------
# Sidebar
# ---------------------------------------

with st.sidebar:

    st.header("About this assistant")

    st.write(
        "This chatbot uses LangChain, "
        "ChromaDB, local embeddings, and "
        "Groq to answer website questions."
    )

    st.write(
        "It cannot book appointments or "
        "provide personalized medical advice."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------
# Example questions
# ---------------------------------------

if not st.session_state.messages:

    st.write("### Example Questions")

    st.markdown(
        """
        - What treatments are available at ITC?
        - What does the adrenal cancer package include?
        - What prices are listed for ITC?
        - What information is available about Tijuana?
        """
    )


# ---------------------------------------
# Display conversation history
# ---------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown("**Sources:**")

            for source in message["sources"]:

                st.markdown(
                    f"- [{source['title']}]"
                    f"({source['url']})"
                )


# ---------------------------------------
# Chat input
# ---------------------------------------

question = st.chat_input(
    "Ask about treatments, packages, or costs..."
)


# ---------------------------------------
# Generate response
# ---------------------------------------

if question:

    # Prior messages, excluding current question.
    history = list(st.session_state.messages)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching approved PlacidWay sources..."
        ):

            response = ask_chatbot(
                question=question,
                history=history
            )

        st.markdown(response["answer"])

        if response["sources"]:

            st.markdown("**Sources:**")

            for source in response["sources"]:

                st.markdown(
                    f"- [{source['title']}]"
                    f"({source['url']})"
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"],
        "status": response["status"]
    })
