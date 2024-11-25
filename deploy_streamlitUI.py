# Set Streamlit page config
st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

# Sidebar Information
st.sidebar.title("🫁 Lung Cancer Detection")
st.sidebar.markdown(
    """
    ### About this App:
    - This application uses a pre-trained EfficientNet-B0 model fine-tuned for lung cancer detection.
    - Upload a CT scan image to get predictions.
    - **Ensure the image is in .jpg or .png format.**

    ### Disclaimer:
    - **This model is for demonstration purposes only.**
    - It is not intended for clinical or diagnostic use.
    - Always consult a qualified healthcare professional for medical advice or diagnoses.
    """
)

# Load the pre-trained model
@st.cache_resource  # Cache model to avoid reloading on each interaction
def load_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 4)  # 4 classes as per your dataset
    )
    model.load_state_dict(torch.load("C:\\Users\\malee\\BESTMODEL.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Main App Section
st.title("🫁 AI-Assisted Detection of Lung Cancer")
st.markdown(
    """
    **Welcome to the AI-assisted lung cancer detection application.**  
    ---
     ### Instructions:
    1. Upload a CT scan image using the file uploader below.
    2. Click **Predict** to see the results.
    
    """
)

# File Uploader
uploaded_file = st.file_uploader("Upload a CT scan image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        # Apply transformations
        input_image = transform(image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(input_image)
            _, prediction = torch.max(output, 1)
        
        # Class Names
        class_names = ['squamous.cell.carcinoma', 'large.cell.carcinoma', 'adenocarcinoma', 'normal']
        
        # Display Results
        st.markdown(
            f"""
            ### Prediction Results:
            - **Predicted Class**: {class_names[prediction.item()]}
            - **Confidence Scores**: {torch.softmax(output, dim=1).numpy()[0]}
            ---
            **Note:** These results are for demonstration purposes only.  
            Always seek professional medical advice for health-related concerns.
            """
        )