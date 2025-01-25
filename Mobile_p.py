import streamlit as st

pages = {
    "MAIN APPLICATION":[
    st.Page('./Mobile_Price/Data.py', title='Data', icon='🔍'),
    st.Page('./Mobile_Price/EDA.py', title='EDA', icon='📈'),
    st.Page('./Mobile_Price/Chart.py', title='Chart', icon='📉'),
    st.Page('./Mobile_Price/Data processing.py', title='Data processing', icon='⚙️'),
    st.Page('./Mobile_Price/Machine Learning model.py', title='Machine Learning Model', icon='🤖'),
    st.Page('./Mobile_Price/Testing.py', title='Testing', icon='📋')
    ],
}

pg = st.navigation(pages)
pg.run()

# Create columns with a ratio to place the buttons on the right
col1, col2 = st.columns([4, 1])

with col2:
    st.markdown("<div style='position: absolute; bottom: 10px; right: 10px;'>", unsafe_allow_html=True)
    
    if st.button("Data", key="Data", help="Go to Data", use_container_width=True):
        st.switch_page("./Mobile_Price/Data.py")
    
    if st.button("EDA", key="eda", help="Go to EDA", use_container_width=True):
        st.switch_page("./Mobile_Price/EDA.py")
    
    if st.button("Chart", key="chart", help="Go to Chart", use_container_width=True):
        st.switch_page("./Mobile_Price/Chart.py")
    
    if st.button("Data processing", key="data_processing", help="Go to Data Processing", use_container_width=True):
        st.switch_page("./Mobile_Price/Data processing.py")
    
    if st.button("Machine Learning model", key="ml_model", help="Go to ML Model", use_container_width=True):
        st.switch_page("./Mobile_Price/Machine Learning model.py")
    
    if st.button("Testing", key="testing", help="Go to Testing", use_container_width=True):
        st.switch_page("./Mobile_Price/Testing.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

if st.sidebar.button("Reset all of the Settings"):
    st.session_state.clear()
    st.switch_page("./Mobile_Price/Data.py")