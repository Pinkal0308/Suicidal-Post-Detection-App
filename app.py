import streamlit as st
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import plotly.express as px
import pandas as pd
import gc
from deep_translator import GoogleTranslator  # For multilingual support

# Function to clear session memory
def clear_memory():
    gc.collect()

# Function to translate text to English (default)
def translate_text(text, source_lang='auto', target_lang='en'):
    try:
        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated_text
    except Exception as e:
        st.error("Translation failed: {}".format(str(e)))
        return text  # Return original text if translation fails

# Load the tokenizer
token_form = pickle.load(open('tokenizer.pkl', 'rb'))

# Clear memory before loading the model
clear_memory()

# Load the model efficiently, freeing up memory
try:
    del model  # Clear any previous model (if any exists in memory)
except:
    pass

model = load_model("model.h5")

if __name__ == '__main__':
    st.title('Suicidal Post Detection App (Multilingual Support)')
    st.subheader("Input the Post content below (Any Language)")

    # Text input for the post
    sentence = st.text_area("Enter your post content here")

    # Dropdown for selecting language (optional)
    language_options = ['auto', 'en', 'fr', 'es', 'de', 'hi', 'zh-cn', 'ru']
    source_lang = st.selectbox("Select the language of the input text", language_options, index=0)

    # Button to predict
    predict_btt = st.button("Predict")
    
    if predict_btt:
        # Clear memory before prediction
        clear_memory()

        # Translate text to English (assuming model is trained on English data)
        translated_sentence = translate_text(sentence, source_lang)

        # Display the original and translated text
        st.write("Original Post: " + sentence)
        if source_lang != 'en':
            st.write("Translated Post: " + translated_sentence)
        
        # Preprocess the translated input
        twt = [translated_sentence]
        twt = token_form.texts_to_sequences(twt)
        twt = pad_sequences(twt, maxlen=50)

        # Predict the ideation
        prediction = model.predict(twt)[0][0]

        # Adjust threshold for prediction
        threshold = 0.2  # Make model more sensitive to suicidal ideation
        
        # Display the raw prediction value for transparency
        st.write(f"Raw prediction score: {prediction:.4f}")

        # Display the prediction result with the adjusted threshold
        if prediction > threshold:
            st.warning("Potential Suicide Post")
        else:
            st.success("Non Suicide Post")
        
        # Prepare probability data for visualization
        class_label = ["Potential Suicide Post", "Non Suicide Post"]
        prob_list = [prediction * 100, 100 - prediction * 100]
        prob_dict = {"Category": class_label, "Probability (%)": prob_list}
        df_prob = pd.DataFrame(prob_dict)

        # Create a pie chart for probability distribution
        fig_pie = px.pie(df_prob, names='Category', values='Probability (%)', 
                         title="Prediction Probability Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Create a bar chart for comparison
        fig_bar = px.bar(df_prob, x='Category', y='Probability (%)', color='Category', 
                         title="Prediction Probability Comparison", text_auto=True)
        model_option = "LSTM + GloVe Embeddings"

        # Update the chart and info message based on prediction
        fig_bar.update_layout(title_text="{} Model - Prediction Probability".format(model_option))
        if prediction > threshold:
            st.info("The {} model predicts a {:.2f}% chance that the post is a Potential Suicide Post compared to {:.2f}% for a Non Suicide Post.".format(model_option, prediction * 100, 100 - prediction * 100))
        else:
            st.info("The {} model predicts a {:.2f}% chance that the post is a Non Suicide Post compared to {:.2f}% for a Potential Suicide Post.".format(model_option, 100 - prediction * 100, prediction * 100))

        # Display the bar chart
        st.plotly_chart(fig_bar, use_container_width=True)

        # Model information and description
        st.subheader("Model Details")
        st.write(f"This app uses a {model_option} for detecting potential suicidal ideation in posts. The model was trained on labeled datasets with pre-trained GloVe embeddings for natural language understanding. The threshold has been adjusted to improve detection sensitivity.")

        # Clear memory after processing
        clear_memory()
