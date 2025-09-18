import os


try:
    import streamlit as st
except Exception:
    st = None


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_openai_client():
    """
    Return an OpenAI client if both a key and the 'openai' package are available.
    Otherwise return None so the app can still run in deterministic mode.
    """
    key = os.getenv("OPENAI_API_KEY")

    # Try Streamlit secrets if available
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY", key)
    except Exception:
        st = None  # streamlit not available (e.g., during CLI pipelines)

    # If no key, return None (app can still run in deterministic mode)
    if not key:
        if st:
            st.info("OPENAI_API_KEY not set; running without LLM features.")
        return None

    # Try to import the package
    try:
        from openai import OpenAI
    except Exception:
        if st:
            st.warning("OpenAI package not installed; running without LLM features.")
        return None

    return OpenAI(api_key=key)