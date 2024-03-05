import gradio as gr
from gradio.components import Textbox
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.process_data import DiseasePrognosisDataset, REVERSE_MAPPING

# Load the pre-trained model and tokenizer
model_dir = "C:/Users\Parul\PycharmProjects\disease-prediction\saved_models"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Initialize your model parameters and optimizer
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

def predict(name, age, address, symptoms):
    
    # Tokenize the input text
    inputs = tokenizer(symptoms, return_tensors="pt", max_length=64, padding="max_length", truncation=True)

    # Make the prediction using the model
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class back to the disease name
    predicted_disease = REVERSE_MAPPING[predicted_class]

    return f"Name: {name}\nAge:{age} years\nAddress: {address}\nPrognosis:{predicted_disease}"

iface = gr.Interface(
    title="<div style='background-color: #2B547E ; padding: 10px; border-radius: 5px;'><span style='color: white; font-weight: bold;'>Disease Prediction Interface</span></div>",
    description="<div style='font-size: 25px;'>Enter your details and symptoms to get a prediction.</div>",
    inputs=[
        Textbox(label="Name"),
        Textbox(label="Age"),
        Textbox(label="Address"),
        Textbox(label="Symptoms"),

    ],
    fn=predict,
    outputs="text",

)


# Set a custom CSS style for the interface
iface.css = """
.container {
    font-family: 'Arial', sans-serif;
    background-color: #BCC6CC ;
    padding: 20px;
    border-radius: 10px;
    max-width: 600px;
    margin: auto;
}

input[type="text"] {
    width: 100%;
    padding: 10px;
    margin-bottom: 10px;
    border: 1px solid #000000;
    border-radius: 5px;
    box-sizing: border-box;
}

button {
    background-color: #E5E4E2;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

button:hover {
    background-color: #E5E4E2;
}

"""

iface.launch()


