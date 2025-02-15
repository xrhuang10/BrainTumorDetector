# BrainTumorDetector

<html>
<head>
    <title>Brain Tumor Classification - README</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
        h1, h2 { color: #333; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🧠 Brain Tumor Classification using Neural Networks</h1>
    <p>A machine learning project that uses a neural network to detect and classify brain tumors into one of four categories: <b>Glioma, Meningioma, No Tumor, or Pituitary</b>.</p>
    
    <h2>📂 Dataset</h2>
    <p>The dataset consists of <b>1300 MRI images</b> sourced from Kaggle. It has been preprocessed to improve model performance.</p>
    
    <h2>🛠️ Technologies Used</h2>
    <ul>
        <li><b>Python</b> - Core programming language</li>
        <li><b>Pandas</b> - Data handling and preprocessing</li>
        <li><b>TensorFlow & Keras</b> - Neural network implementation</li>
        <li><b>Adam Optimizer</b> - Used for optimizing the model</li>
    </ul>
    
    <h2>🧑‍💻 Model Architecture</h2>
    <p>The model consists of a <b>3-layer neural network</b> trained for 50 epochs, achieving an accuracy of <b>96%</b>.</p>
    
    <pre><code>
Model Summary:
- Input Layer
- Hidden Layer 1 (Dense + ReLU Activation)
- Hidden Layer 2 (Dense + ReLU Activation)
- Output Layer (Softmax Activation for classification)
    </code></pre>
    
    <h2>🚀 Installation & Usage</h2>
    <pre><code>
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification

# Install dependencies
pip install -r requirements.txt

# Run the training script
python train.py
    </code></pre>
    
    <h2>📊 Results</h2>
    <p>The model achieves <b>96% accuracy</b> after training on 50 epochs.</p>
    
    <h2>📜 License</h2>
    <p>This project is open-source under the <b>MIT License</b>.</p>
    
    <h2>🤝 Contributing</h2>
    <p>Pull requests and suggestions are welcome! Feel free to fork this repo and contribute.</p>
</body>
</html>
