import textwrap

import numpy as np

import streamlit as st


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert a hex color to an rgba(...) string.

    Supports short (#RGB) and long (#RRGGBB) formats. Returns an rgba
    string using the provided alpha or the original value on error.
    """
    if not hex_color:
        return hex_color
    h = hex_color.lstrip("#")
    try:
        if len(h) == 3:
            r = int(h[0] * 2, 16)
            g = int(h[1] * 2, 16)
            b = int(h[2] * 2, 16)
        elif len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
        else:
            return hex_color
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return hex_color


try:
    from .looks import Eleven_COLORS
except Exception:
    # When running the module as a script (e.g. `streamlit run ...`) relative
    # imports may fail because there's no package context. Fall back to
    # importing the top-level module.
    from looks import Eleven_COLORS


def flight_card(flight, departure_time, aircraft, stand, runway, taxi_time, uid=None):
    """Render a single-line flight card with a left chevron button to toggle details.

    - `uid` is used to create a per-card session-state key so expand/collapse
        persists across reruns.
    """
    key = f"card_expanded_{uid}"
    # Ensure a default session state for this card
    if key not in st.session_state:
        st.session_state[key] = False

    # Layout: tiny left column for chevron button, big column for card
    cols = st.columns([0.4, 11])
    with cols[0]:
        # Use a button to toggle expanded state. Label shows chevron direction.
        btn_key = f"btn_{key}"
        chevron = "▸" if not st.session_state[key] else "▾"
        if st.button(chevron, key=btn_key):
            st.session_state[key] = not st.session_state[key]

    # Build the card HTML; if expanded, include drivers inside same .flight-card
    # so expansion doesn't create a separate box. (Chevron rendered as button
    # in the left column, not inside the card.)

    # Mock SHAP values prepared now in case needed for inline rendering
    try:
        seed = int(uid)
    except Exception:
        seed = 0
    np.random.seed(seed)
    cats = ["Weather", "Traffic", "Distance", "Aircraft"]
    vals = np.random.randn(len(cats)) * 2.0
    # Make zip strictness explicit for linters.
    shap = {c: float(v) for c, v in zip(cats, vals, strict=False)}
    items = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Compose drivers HTML as five aligned segments to match the card above.
    # First segment: drivers title (aligned with Flight Number). Next four
    # segments: 1st..4th drivers aligned with Stand, Runway, Departure, Taxi.
    ord_suffix = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    drivers_line_html = "<div class='flight-line'>"

    # Title occupies the first segment
    drivers_line_html += (
        "<div class='driver-segment'>"
        "<div class='drivers-title'>Key Taxi-Out Time Drivers:</div>"
        "</div>"
    )

    # Place each ordered driver into the next segments (Stand, Runway, Departure, Taxi)
    for i, (cat, val) in enumerate(items, start=1):
        color = Eleven_COLORS["positive"] if val > 0 else Eleven_COLORS["negative"]
        bg = hex_to_rgba(color, alpha=0.16)
        label = ord_suffix.get(i, f"{i}th")
        # Arrow up for positive, down for negative
        arrow = "▲" if val > 0 else "▼"
        drivers_line_html += (
            "<div class='driver-segment'>"
            f"<span class='segment-label'>{label}:</span> "
            f"<span class='driver-highlight' style='background:{bg};'>"
            f"<span class='driver-arrow' style='color:{color};'>{arrow}</span>"
            f"{cat}</span></div>"
        )

    # If there are fewer than 4 drivers, fill remaining segments with empty placeholders
    remaining = 4 - len(items)
    for _ in range(remaining):
        drivers_line_html += "<div class='driver-segment'></div>"

    drivers_line_html += "</div>"

    drivers_line = drivers_line_html

    # Full card HTML assembled from shorter parts to satisfy line-length checks
    parts = [
        '<div class="flight-card">',
        '    <div class="flight-line">',
        (
            f'        <div class="flight-segment">'
            f'<span class="segment-label">Flight Number:</span> '
            f'<span class="segment-value">{flight}</span></div>'
        ),
        (
            f'        <div class="flight-segment">'
            f'<span class="segment-label">Stand:</span> '
            f'<span class="segment-value">{stand}</span></div>'
        ),
        (
            f'        <div class="flight-segment">'
            f'<span class="segment-label">Runway:</span> '
            f'<span class="segment-value">{runway}</span></div>'
        ),
        (
            f'        <div class="flight-segment">'
            f'<span class="segment-label">Departure Time:</span> '
            f'<span class="segment-value">{departure_time}</span></div>'
        ),
        (
            f'        <div class="flight-segment">'
            f'<span class="segment-label">Predicted Taxi-Out Time:</span> '
            f'<span class="segment-value taxi">{taxi_time:.1f} min</span></div>'
        ),
        "    </div>",
        (
            ""
            if not st.session_state[key]
            else f"    <div class='drivers-block'>{drivers_line}</div>"
        ),
        "</div>",
    ]

    full_html = "\n".join(parts)

    with cols[1]:
        st.markdown(textwrap.dedent(full_html).strip(), unsafe_allow_html=True)


def render_header(title: str, logo_path: str | None = None):
    """Render the page header with `title` and optional `logo_path` on the right.

    Uses columns to position the title and logo so Streamlit serves the
    local asset correctly.
    """
    if logo_path:
        # Use Streamlit columns and `st.image` directly to ensure the file is
        # served and avoid HTML img issues. Title left, logo right.
        cols = st.columns([9, 1])
        with cols[0]:
            html_parts = [
                "<div style='display:flex;align-items:center;width:100%;'>",
                f"<div class='app-title'>{title}</div>",
                "<div style='width:12px'></div>",
                (
                    f"<div style='height:8px;flex:1;max-width:720px;"
                    f"background:{Eleven_COLORS['accent']};border-radius:6px;"
                    "margin-left:8px;margin-right:8px;'></div>"
                ),
                "</div>",
            ]
            html = "\n".join(html_parts)
            st.markdown(html, unsafe_allow_html=True)
        with cols[1]:
            try:
                st.image(logo_path, width=200, clamp=True)
            except Exception:
                # If st.image fails, fall back to a small markdown image tag
                img_tag = (
                    f"<img src='{logo_path}' style='height:64px;object-fit:contain;'/>"
                )
                st.markdown(img_tag, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
        <div class="streamlit-header">
            <div class="app-title">{title}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
