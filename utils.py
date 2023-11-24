import os
import numpy as np
from PIL import Image
import streamlit as st


@st.cache_data
def show_icon(emoji: str):
    """
        Shows an emoji as Notion-style page icon
    :param emoji: name of the emoji, i.e. ":balloon:"
    :return:
    """
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True
    )