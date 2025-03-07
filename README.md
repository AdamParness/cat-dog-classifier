# Cat vs Dog Classifier

A web application that classifies images as either cats or dogs using a deep learning model based on MobileNetV2.

## Features

- Upload images through a user-friendly web interface
- Real-time classification of cat and dog images
- Confidence score display
- Responsive design that works on mobile and desktop

## Tech Stack

- **Backend**: FastAPI, TensorFlow, Python
- **Frontend**: HTML, JavaScript, Bootstrap
- **Model**: MobileNetV2 (transfer learning)

## Local Development

### Prerequisites

- Python 3.10+
- TensorFlow 2.x
- FastAPI

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/adamparness/cat-dog-classifier.git
   cd cat-dog-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:8000`

## Deployment

This application can be deployed to various platforms:

- **Render**: Connect your GitHub repository and deploy as a Web Service
- **Fly.io**: Use the provided Dockerfile to deploy with `fly launch`
- **Hugging Face Spaces**: Upload your files to a new Space with the FastAPI template
- **Railway**: Connect your GitHub repository for automatic deployment

## Model Training

The model was trained on the Kaggle Cats and Dogs dataset using transfer learning with MobileNetV2 as the base model. The training script is included in `model.py`.

## License

MIT
