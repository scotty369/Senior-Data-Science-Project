import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img # type: ignore
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

# Load images and data (assuming they are in the same directory as the script)
image_path = './Images'
caption_file = "./captions.txt"

try:
    data = pd.read_csv(caption_file)
    print("Data loaded successfully")
except FileNotFoundError:
    print(f"Error: The file {caption_file} was not found.")
    data = pd.DataFrame()

# Load additional images
validation_image = "Validation.png"
results_image = "Results.png"
example_image = "image.png"
blob_image = "blob.png"

# Load sample captions and word pairings
sample_captions = [
    "A black and brown dog playing with a stick.",
    "A mountain biker is jumping his bike over a rock.",
    "A boy and a girl are riding on a camel in the sand on the beach."
]

word_pairings = {
    "dog": "playing",
    "mountain": "biker",
    "boy": "girl",
    "camel": "sand",
    "stick": "black",
    "rock": "jumping",
    "beach": "riding"
}

st.set_page_config(page_title="Image Captioning Tool", layout="wide")

# Navigation using tabs/pages
pages = {
    "Project Overview": "Project Overview",
    "Project Need": "Project Need",
    "Project Uses": "Project Uses",
    "Model Overview": "Model Overview",
    "Methodology": "Methodology",
    "Results": "Results",
    "Analysis": "Analysis",
    "Conclusion": "Conclusion"
}

page = st.sidebar.radio("Go to", list(pages.keys()))

if page == "Project Overview":
    st.title("Senior Project: AI Image Captioning Tool")
    st.markdown("""
    ## Project Overview
    This senior project introduces an AI-driven tool designed to automatically generate concise and contextually relevant captions for images. By leveraging advanced machine learning and computer vision techniques, the tool aims to identify and describe key visual elements in natural language. This automation not only streamlines image description processes but also enhances content organization, improves user engagement, and significantly increases accessibility for visually impaired users.
    
    The broader applications of this technology extend to automating workflows for professionals across various sectors, including businesses, photographers, and social media managers, thereby optimizing content creation and management.
    """)

elif page == "Project Need":
    st.title("Addressing the Need for Automated Image Description")
    st.markdown("""
    ## Project Need
    In the era of exponentially growing digital image data, the manual tagging and captioning of images has become increasingly challenging and time-consuming. This process is often inconsistent, leading to difficulties in content organization and searchability.
    
    This project addresses this critical need by providing an automated solution that generates accurate and relevant captions. By automating this task, the tool saves substantial time, ensures consistency, and enhances image discoverability, significantly benefiting businesses, photographers, and social media managers in streamlining their workflows and optimizing content creation.
    """)

elif page == "Project Uses":
    st.title("Applications of the Image Captioning Tool")
    st.markdown("""
    ## Project Uses
    The versatility of this tool extends across numerous applications, enhancing efficiency and organization in various fields:
    
    -   **Automated Photo Categorization:** Efficiently tagging and organizing images for easy retrieval.
    -   **Social Media Automation:** Generating compelling and relevant captions for social media posts.
    -   **E-commerce Product Descriptions:** Creating detailed and accurate product descriptions for online retail.
    -   **Integration with AI Models:** Enhancing caption accuracy by integrating with existing AI models.
    
    These applications demonstrate the tool's potential to automate image-related tasks, improve workflow efficiency, and enhance content creation across diverse sectors.
    """)

elif page == "Model Overview":
    st.title("Model Architecture: CNN and LSTM")
    st.markdown("""
    ## Model Overview
    The image captioning tool employs a sophisticated architecture, combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to analyze images and generate detailed, contextually accurate captions.
    
    -   **CNN (DenseNet201):** Extracts essential features from images, including objects, colors, and textures.
    -   **LSTM:** Utilizes these extracted features to generate grammatically correct and meaningful captions.
    
    In essence, the CNN acts as the "visual processor," identifying key elements, while the LSTM functions as the "narrative generator," constructing sentences based on these elements.
    
    ### Explanation of CNN and LSTM
    -   **Convolutional Neural Networks (CNN):** Specialized deep learning models for image recognition. DenseNet201 processes images, detecting patterns, shapes, textures, and objects, and extracting features crucial for caption generation.
    -   **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) designed for sequential data. After CNN feature extraction, the LSTM generates structured captions by predicting subsequent words based on context.
    """)

elif page == "Methodology":
    st.title("Project Methodology")
    st.markdown("""
    ## Data Preparation
    -   **Data Cleaning**: Ensuring dataset integrity by removing errors and inconsistencies.
    -   **Text Preprocessing**: Formatting captions for model training, including tokenization and sequence padding.

    ### Code for Data Preprocessing:
    ```python
    import string
    from nltk.tokenize import word_tokenize

    caption = "A black and brown dog playing with a stick."
    caption = caption.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = word_tokenize(caption)
    print(tokens)
    ```
    This code demonstrates caption cleaning and tokenization using the **NLTK** library.

    ## Feature Extraction
    Using **DenseNet201**, a pre-trained CNN, to extract high-level image features.

    ### Code for Feature Extraction:
    ```python
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.preprocessing import image
    import numpy as np

    model = DenseNet201(weights='imagenet', include_top=False)
    img_path = 'example_image.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array)
    print(features.shape)
    ```
    This code extracts image features using **DenseNet201**, crucial for capturing visual content.

    ## Caption Generation
    Using an **LSTM network** to generate captions based on image features and word sequences.

    ### Code for LSTM Caption Generation:
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Embedding, Dense

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=256))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(5000, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    ```
    This LSTM model generates captions word by word, using embeddings and image features.

    ## Model Training and Evaluation
    -   Model trained on the **Flickr8k dataset**.
    -   Evaluation using metrics like **BLEU score** to assess caption quality.

    ### Code for BLEU Score Calculation:
    ```python
    from nltk.translate.bleu_score import corpus_bleu

    references = [[['a', 'dog', 'playing', 'with', 'a', 'stick']]]
    hypothesis = [['a', 'dog', 'playing', 'with', 'stick']]
    bleu_score = corpus_bleu(references, hypothesis)
    print(f"BLEU Score: {bleu_score}")
    ```
    The **BLEU score** evaluates the overlap between generated and reference captions.
    """)

elif page == "Results":
    st.title("Model Captioning Results")
    st.image(example_image, caption="Example Image")
    st.markdown("""
    **Sample Captions Generated by the Model:**
    """)
    for caption in sample_captions:
        st.write(f"- {caption}")

    st.markdown("""
    ## Model Training Visualization
    ### Training Loss Over Epochs:
    """)
    st.write("This graph illustrates the training and validation loss of a machine learning model over 12 epochs. The x-axis represents the epochs, while the y-axis shows the loss values. The teal line depicts the training loss, which steadily decreases as the model learns from the training data, indicating improvement over time. The orange line represents the validation loss, which initially decreases, suggesting the model is generalizing well to unseen data. However, after around 5 epochs, the validation loss plateaus and fluctuates slightly, indicating that the model's performance on the validation set is no longer improving significantly, and potentially hinting at overfitting. This divergence between training and validation loss suggests that while the model continues to improve on the training data, its ability to generalize to new data is limited after a certain point.")
    st.image(validation_image, caption="Training Loss")

    st.markdown("""
    ### Generated Caption Examples
    """)
    st.markdown("""
    This collage presents a series of image captions generated by an AI model, showcasing its ability to interpret and describe diverse visual scenes. The captions, prefaced with "startseq" and concluded with "endseq," suggest a structured approach to sentence generation. The model appears to identify key subjects and actions, such as "football player in red uniform," "group of people standing on the street," "two dogs running through the grass," and "young boy in blue shirt playing in the water." The descriptions also capture specific attributes like "pink dress," "red shirt," "black shirt," and "white shirt," indicating the model's capacity to recognize colors and clothing. While the captions are generally straightforward and descriptive, they provide a glimpse into the model's attempt to translate visual information into textual narratives.
    """)
    st.image(results_image, caption="Generated Caption Examples")

elif page == "Analysis":
    st.title("Data Analysis")
    st.markdown("""
    ## Word Associations in Captions
    This word cloud visualization represents the most frequent words found within the image captions of a dataset, offering a quick glimpse into the dataset's content and themes. The larger the word, the more frequently it appears, highlighting the dominant subjects and actions. Notably, "man," "woman," "dog," and "water" stand out, suggesting these are common elements depicted in the images. The cloud also reveals a focus on activities ("running," "walking," "playing," "riding") and specific settings ("beach," "mountain," "lake"). Words like "white," "black," and "blue" indicate color descriptions are prevalent, while terms like "child," "people," and "group" suggest a focus on human subjects. Overall, the word cloud provides a concise overview of the dataset's content, emphasizing common objects, actions, and settings that the image captions describe.
    """)
    st.image(blob_image, caption="Word Cloud of Prominent Words")

    st.markdown("""
    ### Initial Word Pairings:
    """)
    for word, association in word_pairings.items():
        st.write(f"- **{word}**: {association}")

    st.markdown("""
    ## Detailed Image and Caption Display
    """)
    def readImage(path, img_size=224):
        img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
        img = np.array(img) / 255.0
        return img
    
    def display_images(df):
        df = df.reset_index(drop=True)
        for i in range(min(15, len(df))):
            col1, col2 = st.columns(2)
            with col1:
                image = readImage(os.path.join(image_path, df.image[i]))
                st.image(image, caption=f"Image {i+1}", use_column_width=True)
            with col2:
                st.write(f"**Caption {i+1}**: {df.caption[i]}")

    if 'data' in locals() and not data.empty:
        display_images(data.sample(15))
    else:
        st.write("Data not loaded or empty.")

elif page == "Conclusion":
    st.title("Conclusion and Future Development")
    st.markdown("""
    ## Conclusion
    This project successfully developed an AI-powered image captioning tool using CNNs and LSTMs, demonstrating its potential to automate image description tasks, enhance content organization, improve user engagement, and increase accessibility.
    
    ## Future Development Needs
    To further enhance the model's capabilities and broaden its applicability, the following areas require focused development:
    
    -   **Generalization Across Diverse Image Types:** Expanding the model's ability to accurately generate captions for a wider range of image types beyond the limitations of the Flickr8k dataset. This would involve incorporating more sophisticated feature extraction techniques and training on a more varied dataset.
    -   **Integration of Contextual Understanding:** Enhancing the model's capacity to understand and incorporate contextual information, such as the relationships between objects in an image and the broader scene, to produce more nuanced and accurate captions.
    -   **Refinement of Language Generation:** Improving the model's language generation capabilities to produce captions that are not only grammatically correct but also more natural and human-like. This could involve exploring advanced language models and incorporating techniques for generating more diverse and creative captions.
    -   **User Interface and Accessibility:** Developing a user-friendly interface that allows for easy interaction with the tool, including features that make it more accessible to users with visual impairments. This could involve incorporating voice input and output, as well as customizable display options.
    
    By addressing these areas, the model can evolve into a more versatile and robust tool, capable of generating accurate and insightful captions for a wide variety of images, thereby significantly enhancing its utility across different applications.
    """)
