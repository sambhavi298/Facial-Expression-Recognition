# Facial Expression Recognition

Facial Expression Recognition using Deep Learning.

## 📦 Repository Structure

- `Video_model/` – Scripts and pre-trained models for video-based facial expression recognition
- `outputs/`     – Output files, prediction results, and logs
- `traning data/` – Image and video datasets for training
- `video_datasets/` – Datasets used for inference or testing

## ⚙️ Installation

git clone https://github.com/sambhavi298/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
pip install -r requirements.txt


## 🚀 Usage

### Training

Place your training data in the `traning data/` folder and run:
python Video_model/train.py --data_dir "traning data/"

### Prediction

Add your videos to `video_datasets/` and run:

Results will be stored in the `outputs/` folder.

## 📝 Example

Example usage of predict.py
from Video_model.predict import predict_expression

result = predict_expression('video_datasets/sample.mp4')
print(result)

## 💡 Features

- Predicts facial expression from video input
- Compatible with major facial expression datasets
- Modular codebase for easy experimentation

## 📑 License

MIT License

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!
