import streamlit as st

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Main"

def main_tab():
    st.title("Main Tab.")
    st.write("MNIST Image Classification with Kafka!")
    
    if st.button("Перейти на Вкладку 1"):
        st.session_state.current_tab = "Tab 1"
    
    if st.button("Перейти на Вкладку 2"):
        st.session_state.current_tab = "Tab 2"

def tab_input_image():
    st.title("Вкладка 1")
    st.write("Это содержимое первой вкладки.")
    
    if st.button("Вернуться на главную"):
        st.session_state.current_tab = "Main"

def tab_2():
    st.title("Вкладка 2")
    st.write("Это содержимое второй вкладки.")

    if st.button("Вернуться на главную"):
        st.session_state.current_tab = "Main"

