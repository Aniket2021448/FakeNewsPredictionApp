import streamlit as st
st.set_page_config(page_title="News Prediction", page_icon=":earth_africa:")
from streamlit_option_menu import option_menu
import About, Home, Contact


def render_About_page():
    About.main()


def render_home_page():
    Home.main()


def render_contact_page():
    Contact.main()




# TO remove streamlit branding and other running animation
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Spinners
bar = st.progress(0)
for i in range(101):
    bar.progress(i)
    # time.sleep(0.02)  # Adjust the sleep time for the desired speed

st.balloons()

# Web content starts
# Navbar starts
    # Create the Streamlit app
col1, col2 = st.columns([1, 10])
with col1:
    st.header("	:globe_with_meridians:")
with col2:
    st.header("Fake News Prediction App")
    


selected = option_menu(
    menu_title="",
    options=["Home", "About", "Contact"],
    icons=['house', 'kanban', 'envelope'],
    menu_icon="",
    default_index=0,
    orientation="horizontal",
    styles="height: {300px;}, padding: {0px;}, margin: {0px;}, background-color: {white;}"
)

if selected == "About":
    render_About_page()
elif selected == "Contact":
    render_contact_page()
else:
    render_home_page()

