"""Application entry point."""

import streamlit as st

from src.grader import GraderError, grade_solution
from src.ui import render_page


def main() -> None:
    st.set_page_config(page_title="MathVision · 几何解答批改", page_icon="📐", layout="wide")
    render_page(grade_solution, GraderError)


if __name__ == "__main__":
    main()
