import streamlit as st

Eleven_COLORS = {
    "primary": "#FFFFFF",
    "secondary": "#333333",
    "accent": "#4FB583",
    "accent_light": "#DBF0E6",
    "text": "#3C3C3C",
    "text_muted": "#6B6B6B",
    "positive": "#41D348",
    "negative": "#E31D1D",
    "border": "#E0E0E0",
}


# Backwards-compatible primary color alias
PRIMARY_COLOR = Eleven_COLORS["accent"]


def get_color(key: str) -> str:
    """Return the hex color for `key` from the Eleven_COLORS palette.

    Returns None if the key is not present.
    """
    return Eleven_COLORS.get(key)


def apply_page_style():
    """Inject global CSS styles for the Streamlit app.

    Uses colors from the `Eleven_COLORS` palette to theme headers, cards,
    and other components.
    """
    st.markdown(
        f"""
    <style>
    .streamlit-header {{
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 12px;
        margin-bottom: 18px;
    }}

    .app-title {{
        font-size: 32px;
        font-weight: 800;
        color: {Eleven_COLORS["accent"]};
        margin: 0;
    }}

    .kpi-row {{
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }}

    .kpi-card {{
        background: {Eleven_COLORS["primary"]};
        border: 1px solid {Eleven_COLORS["border"]};
        border-radius: 12px;
        padding: 14px;
        min-width: 160px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }}

    .kpi-label {{
        color: {Eleven_COLORS["text_muted"]};
        font-size: 12px;
        margin-bottom: 6px;
    }}

    .kpi-value {{
        color: {Eleven_COLORS["text"]};
        font-size: 22px;
        font-weight: 700;
    }}

    /* Simplified single-line card with thin green vertical separators */
    .flight-card {{
        background: {Eleven_COLORS["primary"]};
        border-radius: 10px;
        padding: 10px 14px 12px 40px; /* extra left padding to fit chevron */
        margin-bottom: 12px;
        border: 2px solid #222; /* black outline */
        font-family: 'Inter', sans-serif;
    }}

    .flight-line {{
        display:flex;
        align-items:center;
        gap:0;
        width:100%;
        justify-content:stretch;
    }}

    .flight-segment {{
        display:flex;
        align-items:center;
        gap:8px;
        padding:0 14px;
        white-space:nowrap;
        flex:1 1 0; /* allow segments to grow and fill the card */
        justify-content:center;
    }}

    /* Thin vertical separator in accent color between segments */
    .flight-segment + .flight-segment {{
        border-left: 2px solid {Eleven_COLORS["accent"]};
        margin-left:6px;
        padding-left:14px;
    }}

    .segment-label {{
        color: {Eleven_COLORS["text_muted"]};
        font-size:13px;
        font-weight:600;
        margin-right:6px;
    }}

    .segment-value {{
        color: {Eleven_COLORS["secondary"]};
        font-size:14px;
        font-weight:700;
    }}

    .segment-value.taxi {{
        color: {Eleven_COLORS["accent"]};
        font-weight:800;
    }}

    /* Driver segments (inside expanded area) should not show separators */
    .driver-segment {{
        display:flex;
        align-items:center;
        gap:8px;
        padding:0 14px;
        white-space:nowrap;
        flex:1 1 0;
        justify-content:center;
    }}

    .driver-line {{
        display:flex;
        gap:0;
        width:100%;
        align-items:center;
    }}

    .driver-highlight {{
        padding:4px 8px;
        border-radius:6px;
        color: #000; /* black text over the colored highlight */
        font-weight:400;
        display:inline-block;
    }}

    .driver-arrow {{
        font-size:12px;
        line-height:1;
        margin-right:6px;
        vertical-align:middle;
    }}

        .drivers-block {{
            margin-top: 16px; /* more vertical separation between card and drivers */
        }}

    .drivers-title {{
        color: {Eleven_COLORS["secondary"]};
        font-size: 13px; /* slightly smaller than main headings */
        font-weight: 600;
            margin: 0 12px 0 0;
            display:inline-flex;
            align-items:center;
    }}

    /* visual chevron removed; chevron button is rendered outside the card */

    .small-muted {{
        color: {Eleven_COLORS["text_muted"]};
        font-size: 12px;
    }}

    /* Page background */
    html, body, .stApp, .main, .block-container {{
        background: {Eleven_COLORS["primary"]};
    }}

    /* Reduce side margins and cap content width */
    .block-container {{
        max-width: 1400px;
        padding-left: 24px;
        padding-right: 24px;
        margin-left: auto;
        margin-right: auto;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )
