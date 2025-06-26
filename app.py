import streamlit as st
import joblib

st.title('Movie Review Sentiment Analyzer')
st.write('Movie review sentiment analysis with Multinomial Naive Bayes')

@st.cache_data
def load_model():
    '''loads the pre-trained model and target names.'''
    model = joblib.load('sentiment_analysis.pkl')
    return model

model = load_model()
text_input = st.text_area('Input a movie review for analysis.')

analyze = st.button('Analyze')

if analyze:
    prediction = model.predict([text_input])
    prediction_proba = model.predict_proba([text_input])
    if prediction[0] == 'positive':
        format_proba = prediction_proba[0][1]
        st.header(':tada:')
    else:
        format_proba = prediction_proba[0][0]
        st.header(':bomb:')

    format_proba = int(round(100*format_proba,0))
    st.subheader('Prediction:')

    st.write(f'This review is predicted to be {prediction[0]} with {format_proba}% confidence.')