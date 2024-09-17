import streamlit as st
import numpy as np
from PIL import Image,ImageDraw
from keras.models import load_model
import tensorflow as tf


from mtcnn import MTCNN
import os
st.set_page_config(
    layout="wide",
    page_title="Deep Fake Image Classifier",
    page_icon=":detective:",
    initial_sidebar_state="expanded",
    )

model=load_model('./new_model.h5')

st.title('DeepFake Image Classifier')


intro_text = """
<div style="font-family: Arial, sans-serif;">
    <h2>Welcome to the Deep Fake Image Classification Web Application!</h2>
    <p>Our platform utilizes cutting-edge artificial intelligence algorithms to tackle the growing challenges posed by deep fake technology. With the proliferation of deep fake images across various online platforms, it has become increasingly difficult to discern between authentic and manipulated content. Our app aims to address this issue by providing users with a reliable tool to detect and classify deep fake images with precision.</p>
    <p>Powered by state-of-the-art machine learning models, our Deep Fake Image Classifier analyzes image data to identify subtle discrepancies and anomalies indicative of manipulation. Whether you're verifying the authenticity of an image, a social media user concerned about fake content. our application offers a user-friendly interface and best possible results.</p>
</div>
"""


st.markdown(intro_text, unsafe_allow_html=True)
# Define the main function to create the web app
def main():
    
     
    
    def preprocess_image(image_args):
        image_new = image_args.resize((224, 224))
        IMage = np.array(image_new) # Normalize pixel values to [0, 1]
        return IMage

    # Function to classify image
    def classify_image(image_par):
    
        image_newly = preprocess_image(image_par)
        image_newly_1 = np.expand_dims(image_newly, axis=0)  # Add batch dimension
        predictions = model.predict(image_newly_1)
        return predictions


    # Main content
    st.markdown("""
    <style>
    .content {
        padding: 20px;
    }
    .content h2 {
        margin-bottom: 10px;
    }
    .content form input[type="file"] {
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    
     # Show some description about deep fake images
    st.write("---")
    writes= """
    # About Deep Fakes

    Deep fakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness.
     These are typically generated using deep learning techniques like generative adversarial networks (GANs).
    They can be used for entertainment purposes, but also pose risks in terms of misinformation and privacy.
    
    """
    st.markdown(writes)
    
    col3,col4=st.columns(2)
    
    with col3:
        st.write("The image below shows that how deepfakes have grown exponentially with years.")

        st.image("./deepfake-videos-growth.jpg",caption='Growth',width=600)
    with col4:
        st.write("Overview of Deepfakes Creation")    
        st.image("./How-Does-Deepfake-Work.png",caption='Creation',width=600)
    # About Section
    st.write("---")
    how_it_works_text = """
    # How This App Works

    ## Overview
    The Deep Fake Image Classification web application facilitates the process of image classification through a systematic procedure. Initially, users are prompted to upload an image via the designated upload section. Upon image submission, the application employs the Multi-Task Cascaded Convolutional Neural Network (MTCNN) model to detect the presence of human faces within the uploaded image. This step ensures that the subsequent classification process is performed exclusively on facial regions, thereby enhancing the accuracy and specificity of the classification outcome.

    ## Facial Feature Detection
    Following the successful detection of facial features within the uploaded image, the identified facial region is extracted and presented to the classification model. Utilizing sophisticated machine learning algorithms, the classification model meticulously analyzes the facial attributes and features present within the extracted region. Leveraging a comprehensive dataset and advanced neural network architectures, the model discerns intricate patterns and nuances characteristic of both authentic and deep fake images.

    ## Classification Process
    Through this meticulous analysis, the model generates a prediction regarding the authenticity of the facial image, distinguishing between genuine representations and those manipulated by deep fake technology. The classification outcome is subsequently presented to the user, providing valuable insights into the authenticity of the uploaded image. By seamlessly integrating image detection and classification functionalities, the Deep Fake Image Classification web application offers users a reliable and efficient solution for discerning genuine imagery from deep fake manipulations.
    """

    # Render the text with h2 tags using Streamlit markdown component
    st.markdown(how_it_works_text)
    # Features Section
    st.write("---")
    features= """
    # Features

    ## Test Accuracy

    The model demonstrates a test accuracy of 71.33%, showcasing its ability to accurately classify images as authentic or manipulated when evaluated on unseen data. This metric is essential for assessing the model's performance in real-world scenarios, indicating its reliability in detecting deep fake images outside of the training dataset.
    
    ## Dataset
    
    Utilizes the Faces 224x224 dataset sourced from Kaggle.
    The dataset contains images of faces resized to 224x224 pixels, enabling efficient training and classification.
    ## Improvements

    This is a just a beginner project which will need improvements in the future but can be used to identify fake images created using AI.
    """
    st.markdown(features)
    # Classification Section
    st.write("---")
    st.markdown("""# Checkout some examples to know more""")
    st.write("User can select the examples from below to check how the model works")
   
    # Pre-load example images
    example_images_folder = "example_images"
    example_images=[os.path.join(example_images_folder, img) for img in os.listdir(example_images_folder)]
    
    # Dropdown menu to select example image
    
    selected_image_path = st.selectbox("Select an example image:",example_images)

# Display the selected image
    
    ex_image = Image.open(selected_image_path)
    st.image(ex_image,width=300)
    if st.button('check'):
           
            try:   
                with st.spinner('Detecting faces...'):
                    Detector=MTCNN()
                    picture = Image.open(selected_image_path)    
                    mage=np.array(picture)
                    output=Detector.detect_faces(mage)
                    if ((output[0]['confidence'])*100)<90:
                        st.write("confidence level:",output[0]['confidence'])
                        raise Exception
                        
            
                    else:
                        for i in output:
                            x,y,width,hieght=i['box']
                            img1 = ImageDraw.Draw(picture)   
                            x1, y1 = x,y # Top-left corner
                            x2, y2 = x+width,y+hieght  # Bottom-right corner


                                
                            crop_pic = picture.crop( (x1, y1 ,x2, y2) )  
                            st.image(crop_pic,caption="face_detected",width=300)      
                       
                
                
                            with st.spinner('Classifying...'): 
                                    prediction = classify_image(crop_pic)
                                    if prediction >= 0.5:
                                        st.error(f"{(prediction[0][0])*100:.2f} % Fake")
                                        
                                    elif prediction <0.5:
                                        st.success(f"{100.0-(prediction[0][0])*100:.2f} % Real")
                    
            except Exception as e:
                st.error(e)



    st.markdown("""# Classification""")
    uploaded_image=st.file_uploader("Choose an Image",type=['jpeg','jpg','png'])
    if uploaded_image is not None:
        col1,col2=st.columns(2)
        try:  
            with col1:
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image',width=300)
                

            if st.button('Classify'):
                try:
                    
                    with st.spinner('Detecting faces...'):
                        detector=MTCNN()
                        picture = Image.open(uploaded_image)    
                        mage=np.array(picture)
                        output=detector.detect_faces(mage)
                        if ((output[0]['confidence'])*100)<90:
                            st.write("confidence level:",output[0]['confidence'])
                            raise Exception
                
                        else:
                            for i in output:
                               
                                    x,y,width,hieght=i['box']
                                    img1 = ImageDraw.Draw(picture)   
                                    x1, y1 = x,y # Top-left corner
                                    x2, y2 = x+width,y+hieght  # Bottom-right corner


                                    img1.rectangle([(x1, y1), (x2, y2)], outline="white")     
                                    crop_pic = picture.crop( (x1, y1 ,x2, y2) )  
                                    st.image(crop_pic,caption="face_detected",width=300)      
                                # st.write("confidence level:",output[0]['confidence'])
                        
                        
                                    with st.spinner('Classifying...'): 
                                        prediction = classify_image(crop_pic)
                                        if prediction > 0.5:
                                            st.error(f"{(prediction[0][0])*100:.2f} % Fake")
                                            
                                        elif prediction <=0.5:
                                            st.success(f"{100.0-(prediction[0][0])*100:.2f} % Real")
                        
                except:
                    st.error("Failed to Detect Face")
        
        
                
        except :
         st.write("Something Went Wrong !!!")

    # FAQ Section
    def display_faqs():
            st.header("Frequently Asked Questions (FAQs)")

            # FAQ 1
            with st.expander("What is a deep fake image?"):
                st.write("A deep fake image is a manipulated image created using artificial intelligence (AI) and machine learning techniques. These images can alter facial features, expressions, or entire scenes to create realistic but fabricated content.")

            # FAQ 2
            with st.expander("How does the Deep Fake Image Classifier work?"):
                st.write("The Deep Fake Image Classifier uses machine learning algorithms to analyze images and detect patterns indicative of deep fake manipulation. It compares visual cues and features within the image to distinguish between authentic and manipulated content.")

            # FAQ 3
            with st.expander("What types of deep fake manipulations can the classifier detect?"):
                st.write("At its beginner stage, the classifier focuses on detecting basic manipulations commonly associated with deep fake images. These may include facial swaps, simple alterations to facial expressions, or other visible anomalies within the image.")

            # FAQ 4
            with st.expander("How accurate is the Deep Fake Image Classifier at this stage?"):
                st.write("As the application is in its early stages, its accuracy may vary and may not be as precise as more advanced models. However, efforts are being made to improve its accuracy through ongoing development and refinement of the classification algorithms.")

            # FAQ 5
            with st.expander("Who can benefit from using the Deep Fake Image Classifier?"):
                st.write("The Deep Fake Image Classifier can benefit users who are concerned about the authenticity of images circulating online. It provides a basic tool for verifying the credibility of images and raising awareness about the prevalence of deep fake manipulation.")

            # FAQ 6
            with st.expander("Is the Deep Fake Image Classifier accessible to the public?"):
                st.write("Yes, the Deep Fake Image Classifier is available for public use as a web application. Users can upload images and receive classification results to determine whether the images contain signs of deep fake manipulation.")

            # FAQ 7
            with st.expander("What are the limitations of the Deep Fake Image Classifier in its current state?"):
                st.write("As a beginner stage application, the classifier may have limitations in detecting complex or subtle manipulations within images. It may also lack certain features and functionalities present in more advanced deep fake detection systems.")
    st.write("---")
    display_faqs()

    
# Run the main function
if __name__ == '__main__':
    main()



