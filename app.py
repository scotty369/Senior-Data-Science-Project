import streamlit as st

st.set_page_config(page_title="Image Captioning Tool", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Methodology", "Results", "Conclusion"])

if page == "Introduction":
    st.title("Senior Project: Image Captioning Tool")
    st.markdown("""
    ## Introduction
    The rapid advancement of artificial intelligence has unlocked exciting possibilities in the realm of computer vision, enabling machines to "see" and interpret images in ways that were previously unimaginable. This project delves into the fascinating field of image captioning, aiming to develop a sophisticated tool that can automatically generate accurate and descriptive captions for images.
    
    This project leverages the power of deep learning, specifically Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, to build an image captioning model. The model will be trained on the Flickr8k dataset, a rich collection of images paired with human-written captions.
    """)

elif page == "Methodology":
    st.title("Project Methodology")
    st.markdown("""
    ## Data Preparation
    - **Data Cleaning**: Ensuring the dataset is free of errors and inconsistencies.
    - **Text Preprocessing**: Formatting captions for model training.

    ### Code for Data Preprocessing:
    Here’s how we process captions and prepare them for training:

    ```python
    import string
    from nltk.tokenize import word_tokenize

    # Sample caption
    caption = "A black and brown dog playing with a stick."

    # Clean the caption by removing punctuation and converting to lowercase
    caption = caption.translate(str.maketrans('', '', string.punctuation)).lower()

    # Tokenize the caption
    tokens = word_tokenize(caption)
    print(tokens)
    ```
    This code cleans and tokenizes captions, which is essential for training a deep learning model. We use the **NLTK** library for tokenization.

    ## Feature Extraction
    Using **DenseNet201**, a pre-trained CNN, to extract meaningful features from images.

    ### Code for Feature Extraction:

    ```python
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.preprocessing import image
    import numpy as np

    # Load pre-trained DenseNet model
    model = DenseNet201(weights='imagenet', include_top=False)

    # Load and preprocess the image
    img_path = 'example_image.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features from the image
    features = model.predict(img_array)
    print(features.shape)
    ```
    This code extracts features from an image using the pre-trained **DenseNet201** model, which is a powerful CNN used for feature extraction.

    ## Caption Generation
    An **LSTM network** generates captions by predicting the next word in a sequence based on image features and previous words.

    ### Code for LSTM Caption Generation:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Embedding, Dense

    # Build an LSTM model for caption generation
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=256))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(5000, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # The LSTM model is trained on the features and captions.
    ```
    The **LSTM network** is trained to generate captions word by word. We use embeddings for words and feed the features from the CNN into the LSTM to generate the caption sequence.

    ## Model Training and Evaluation
    - The model is trained on the **Flickr8k dataset**.
    - Evaluation metrics such as **BLEU score** are used to assess performance.

    ### Code for BLEU Score Calculation:
    ```python
    from nltk.translate.bleu_score import corpus_bleu

    # Example of BLEU score evaluation
    references = [[['a', 'dog', 'playing', 'with', 'a', 'stick']]]  # list of reference captions
    hypothesis = [['a', 'dog', 'playing', 'with', 'stick']]  # model-generated caption

    bleu_score = corpus_bleu(references, hypothesis)
    print(f"BLEU Score: {bleu_score}")
    ```
    The **BLEU score** is an evaluation metric for text generation tasks. It compares the generated caption to reference captions and gives a score based on overlap.

    """)

elif page == "Results":
    st.title("Model Captioning Results")
    st.image("image.png", caption="Example Image")
    st.markdown("""
    **Sample Captions Generated by the Model:**
    - A black and brown dog playing with a stick.
    - A mountain biker is jumping his bike over a rock.
    - A boy and a girl are riding on a camel in the sand on the beach.

    ## Model Training Visualization
    ### Training Loss Over Epochs:
    st.image("Validation.png", caption="Training Loss")

    ### Evaluation Metrics:
    st.image("Results.png", caption="Evaluation Results")

    These visualizations show the progress of the model during training, including loss reduction and the final evaluation results.
    """)

elif page == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    This project aims to develop a practical image captioning tool leveraging **CNNs and LSTMs**. It has potential applications in accessibility, image tagging, and creative expression. The insights gained contribute to advancements in AI-driven image understanding.

    ---
    
    *For more details, visit the official documentation or contact the project team.*
    """)
