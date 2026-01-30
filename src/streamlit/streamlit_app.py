import os

import streamlit as st

st.set_page_config(page_title="Taxi-Out Time Prediction Dashboard", layout="wide")

try:
    from .components import flight_card, render_header
    from .data_loading import add_placeholder_predictions, load_flights
    from .looks import apply_page_style
except Exception:
    # Running as a script (streamlit run ...) may not provide a package
    # context for relative imports; fall back to top-level imports.
    from components import flight_card, render_header
    from data_loading import add_placeholder_predictions, load_flights
    from looks import apply_page_style


def main():
    """Main entrypoint for the Streamlit app.

    Applies global styles, renders the header, loads sample data, and
    displays up to 10 flight cards with interactive details.
    """
    apply_page_style()

    # Header (logo placed at repo assets/eleven.png if present)
    logo_path = "assets/eleven.png"
    if not os.path.exists(logo_path):
        logo_path = None
    render_header("Taxi-Out Time Demo Dashboard", logo_path=logo_path)

    df = add_placeholder_predictions(load_flights())

    # Render up to 10 vertical full-width cards (single column). No global KPIs yet.
    for idx, row in df.head(10).iterrows():
        flight = row

        # Render the full-width flight card (visible by default). Pass loop
        # index as `uid` so expansion state is unique per row.
        flight_card(
            flight=flight["Flight"],
            departure_time=flight["Departure time"].strftime("%H:%M"),
            aircraft=flight["Aircraft"],
            stand=flight["Stand"],
            runway=flight["Runway"],
            taxi_time=flight["Predicted taxi time (min)"],
            uid=idx,
        )

        # Details/drilldown removed per redesign; clickable interaction will be added.


if __name__ == "__main__":
    main()
