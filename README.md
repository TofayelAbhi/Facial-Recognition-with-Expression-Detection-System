# Facial Expression Recognition System

This project is a deep learning-based system that recognizes four basic facial expressions — **Angry, Happy, Neutral**, and **Surprised** — from images. It was developed using a custom dataset collected from Bangladeshi volunteers and trained using the ResNet50 model.

## Features

- Real-time face detection using webcam
- Image preprocessing with data augmentation
- Deep learning model using ResNet50 (transfer learning)
- Emotion classification into 4 categories
- Custom dataset creation script
- Evaluation using accuracy and confusion matrix

## Motivation

Facial expressions are a non-verbal way humans communicate emotions. Recognizing these expressions can enhance interactions in:
- Healthcare (detect discomfort in patients)
- Education (track student engagement)
- Security (identify suspicious behavior)
- Human-computer interaction (improve user experience)

## Dataset

- **Source**: 12 local volunteers
- **Classes**: Angry, Happy, Neutral, Surprised
- **Images**: 100 per class per individual (4,000 total)
- **Split**: 10 people for training, 2 for testing

> Note: "Sad" class was removed due to poor distinction from "Neutral".

## Tools and Technologies

| Tool               | Purpose                             |
|--------------------|-------------------------------------|
| Python             | Main programming language           |
| OpenCV + CVZone    | Webcam and face detection           |
| TensorFlow & Keras | Model building and training         |
| Google Colab       | Training with GPU support           |
| Matplotlib         | Visualization of results            |

## Results

- Best accuracy: **76%** (after tuning ResNet50)
- Overfitting occurred when all layers were unfrozen
- Best performance on **Happy** and **Neutral**
- Harder to classify **Angry** and **Surprised**

## Limitations

- Small dataset (only 12 people, indoor setup)
- Only 4 emotion classes
- Requires internet/GPU for training
- Evaluation limited to accuracy and confusion matrix

## Future Improvements

- Add more diverse data (age, gender, lighting, accessories)
- Include more emotions (Sad, Fear, Disgust)
- Implement real-time emotion detection
- Test alternative models like EfficientNet or MobileNet
- Use advanced evaluation metrics (precision, recall, F1-score)

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/facial-expression-recognition.git
   cd facial-expression-recognition
