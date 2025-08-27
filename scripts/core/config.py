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
    key = None
    if st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            pass
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    # Lazy import so missing package doesn't crash startup
    try:
        from openai import OpenAI
    except Exception:
        if st is not None:
            st.warning("OpenAI package not installed; running without LLM features.")
        return None

    return OpenAI(api_key=key)