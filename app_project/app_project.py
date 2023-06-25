import streamlit as st
import pandas as pd
import numpy as np
import sagemaker
import os, io, boto3, json, uuid
from ast import literal_eval
from deployment import get_approved_package, get_model_accuracy

sagemaker_client = boto3.client('sagemaker-runtime')

s3_client = boto3.client('s3')

sagemaker_session = sagemaker.Session()
account_id = sagemaker_session.account_id()

image_endpoint_name = f'image-model-endpoint' # This variable would ideally be brought in via automation
text_endpoint_name = f'text-model-endpoint' # This variable would ideally be brought in via automation

image_model_accuracy = 0.5 # This variable would ideally be brought in via automation
text_model_accuracy = 0.8 # This variable would ideally be brought in via automation
top_n_predictions_displayed = 5 # This variable would ideally be brought in via automation (through a ops-editable config file) 

# Setting Streamlit App layout/dimensions
st.set_page_config(layout = 'wide')
row1_1, row1_2 = st.columns(2)
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Category Mapping - Category : Index 
cat_json = './category_map.json'
with open(cat_json) as file:
    cat_dict = json.load(file)
    cat_dict = dict((v, k) for k, v in cat_dict.items())
    
if 'inf_button' not in st.session_state:
    st.session_state.inf_button = True
    
if 'inf_session' not in st.session_state:
    st.session_state.inf_session = False
    
if 'fbk_session' not in st.session_state:
    st.session_state.fbk_session = False
    
# -----------------------------------------------------------------------------------------

def get_user_input_image():
    
    uploaded_image = col1.file_uploader('Required: Upload a Product Image', type = ['jpg', 'jpeg', 'png'])
    
    uploaded_image_path = ''

    if (uploaded_image is not None):
        
        uploaded_image_path = os.path.join('uploaded_images', uploaded_image.name)
        
        with open(uploaded_image_path, "wb") as user_image:
            
            user_image.write(uploaded_image.getbuffer())

    return uploaded_image_path

def get_user_input_text():
    
    uploaded_text = col1.text_input('Required: Write a Product Title/Description')

    return uploaded_text

# -----------------------------------------------------------------------------------------

@st.cache_data(persist = True)
def categorize_image(image_path_input):
    
    with open(image_path_input, 'rb') as f:
        
        image_payload = f.read()
        image_payload = bytearray(image_payload)
    
    image_response = sagemaker_client.invoke_endpoint(
        
        EndpointName = image_endpoint_name,
        ContentType = 'application/x-image',
        Body = image_payload
    
    )

    image_predictions_list = literal_eval(image_response['Body'].read().decode('utf-8'))
        
    return image_predictions_list

@st.cache_data(persist = True)
def categorize_text(text_input):
    
    text_payload = text_input.encode('utf-8')
        
    text_response = sagemaker_client.invoke_endpoint(
        
        EndpointName = text_endpoint_name,
        ContentType = 'text/csv',
        Body = text_payload
        
    )

    text_predictions_list = json.loads(text_response['Body'].read())['predictions'][0]
        
    return text_predictions_list

# -----------------------------------------------------------------------------------------

@st.cache_data(persist = True)
def process_output(image_pred, text_pred):
    
    predictions_array = []
    
    for i in range(len(text_pred)):
        
        category_label = cat_dict[i]
        
        # category_probability = image_pred[i] * image_model_accuracy + text_pred[i] * text_model_accuracy
        
        category_probability = text_pred[i]
        
        predictions_array.append((category_label, category_probability))
        
        predictions_array.sort(key = lambda tup: tup[1], reverse = True)
        
    return predictions_array

# -----------------------------------------------------------------------------------------

@st.cache_data(persist = True)
def save_inputs_outputs_to_s3(inference_id, text_pred, image_pred, top_n_predictions_array, user_input_text, user_input_image_path):
    
    submit_dict = {
                   'inference_id' : inference_id,
                   'text_pred' : text_pred,
                   'image_pred' : image_pred,
                   'top_n_predictions_array' : top_n_predictions_array,
                   'user_input_text' : user_input_text
                  }
    
    with open(user_input_image_path, 'rb') as image:
        
        user_input_image = image.read()
    
    bucket_name = 'sagemaker-us-east-1-233328792017' # automate
    
    json_file_name = f'{inference_id}-inference.json'
    
    image_file_name = f'{inference_id}-image.jpg'

    s3_client.put_object(Body = json.dumps(submit_dict), Bucket = bucket_name, Key = f'project_inference_saving/model_outputs/{json_file_name}')
    s3_client.put_object(Body = user_input_image, Bucket = bucket_name, Key = f'project_inference_saving/images/{image_file_name}')

@st.cache_data(persist = True)
def submit_user_feedback(inference_id, user_feedback, category_selection):
    
    submit_dict = {
                   'inference_id' : inference_id,
                   'feedback' : user_feedback,
                   'category_selection' : category_selection
                  }
    
    bucket_name = 'sagemaker-us-east-1-233328792017' # automate
    
    feedback_file_name = f'{inference_id}-feedback.json'

    s3_client.put_object(Body = json.dumps(submit_dict), Bucket = bucket_name, Key = f'project_inference_saving/feedback/{feedback_file_name}')

@st.cache_data(persist = True)
def save_user_feedback_to_s3(inference_id, user_feedback, category_selection):
    
    if user_feedback != '-':
        
        submit_user_feedback(inference_id, user_feedback, category_selection)

    else:

        st.write('')
        
# -----------------------------------------------------------------------------------------

def main():
    
    row1_2.title('CS611 Project - e-Commerce Product Classifier (based on images and text)')
    
    row1_1.title(f'User Inputs:')
    
    input_image_path = get_user_input_image()
    
    input_text = get_user_input_text()
    
    if input_image_path != '':
        
        col2.write(f'Uploaded Product Image:')
        
        imageLocation = col2.empty()
        
        imageLocation.image(input_image_path, use_column_width = 'auto')
        
    else:
        
        imageLocation = col2.empty()
        
    if (input_text is not None) and (input_text != ''):
        
        col2.write(f'Uploaded Product Text/Description: {input_text}')
        
    if (input_text is not None) and (input_text != '') and (input_image_path != ''):
        
        st.session_state.inf_button = False
        
    elif ((input_text is None) or (input_text == '')) or (input_image_path == ''):
        
        st.session_state.inf_button = True
        
        st.session_state.inf_session = False
        
        st.session_state.fbk_session = False
    
    inference_button = st.button('Make Inference', disabled = st.session_state.inf_button)
    
    if st.session_state.inf_session and not inference_button:
        
        inference_button = True
    
    if inference_button and (not st.session_state.inf_session or not st.session_state.fbk_session):
        
        st.session_state.inf_session = True

        text_pred, image_pred = categorize_text(input_text), categorize_image(input_image_path)

        top_n_predictions_array = process_output(image_pred, text_pred)

        top_n_predictions_df = pd.DataFrame(top_n_predictions_array[:top_n_predictions_displayed], columns = ['Category', 'Probability'])
        
        top_after_n_predictions_df = pd.DataFrame(top_n_predictions_array[top_n_predictions_displayed:], columns = ['Category', 'Probability'])

        st.write(f'Categories Predicted:')

        st.dataframe(top_n_predictions_df)
        
        inference_id = str(uuid.uuid4())

        user_feedback = st.selectbox('Did you find the predictions accurate?', ('-', 'Yes', 'No'))
        
        if user_feedback == 'Yes':
            
            top_n_predictions_array_feedback = ['-'] + list(top_n_predictions_df['Category'])
            
            category_selection = st.selectbox('Which category did you select?', tuple(top_n_predictions_array_feedback))
            
        elif user_feedback == 'No':
            
            top_n_predictions_array_feedback = ['-', 'Others'] + list(top_after_n_predictions_df['Category'])
            
            category_selection = st.selectbox('Which is the correct category?', tuple(top_n_predictions_array_feedback))
            
        if 'category_selection' in locals():
            
            if (not st.session_state.fbk_session) and category_selection != '-':

                st.session_state.fbk_session = True
                
                save_inputs_outputs_to_s3(inference_id, text_pred, image_pred, top_n_predictions_array, input_text, input_image_path)

                save_user_feedback_to_s3(inference_id, user_feedback, category_selection) 
                
                st.write('Thank you for submitting feedback!')
        
    elif inference_button and st.session_state.inf_session and st.session_state.fbk_session:

        text_pred, image_pred = categorize_text(input_text), categorize_image(input_image_path)

        inference_id = str(uuid.uuid4())

        top_n_predictions_array = process_output(image_pred, text_pred)

        top_n_predictions_df = pd.DataFrame(top_n_predictions_array[:top_n_predictions_displayed], columns = ['Category', 'Probability'])

        st.write(f'Categories Predicted:')

        st.dataframe(top_n_predictions_df)

        user_feedback = st.selectbox('Did you find the predictions accurate?', ('-', 'Yes', 'No'))
        
        if user_feedback == 'Yes':
            
            top_n_predictions_array_feedback = ['-'] + list(top_n_predictions_df['Category'])
            
            category_selection = st.selectbox('Which category did you select?', tuple(top_n_predictions_array_feedback))
            
        elif user_feedback == 'No':
            
            top_n_predictions_array_feedback = ['-'] + [cat[0] for cat in top_n_predictions_array]
            
            category_selection = st.selectbox('Which category did you select?', tuple(top_n_predictions_array_feedback))
        
        st.write('Thank you for submitting feedback!')
            
if __name__ == '__main__':
    
    main()
