import os
from openai import OpenAI


try:
    import streamlit as st
except Exception:
    st = None


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_openai_client() -> OpenAI | None:
    """
    Return an OpenAI client if a key is available, otherwise None.
    This must NOT raise so the app can still boot without an API key.
    """
    key = None

    if st is not None:
        try:
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None

    # Fallback to environment variable (local dev)
    key = key or os.getenv("OPENAI_API_KEY")

    if not key:
        return None  
    
    try:
        from openai import OpenAI
    except Exception:
        return None

    return OpenAI(api_key=key)
