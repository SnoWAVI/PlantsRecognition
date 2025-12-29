<div align="center"><h1>🌿 Plants Recognition</h1></div>
<div align="center"><img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"> <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"> <img src="https://img.shields.io/badge/CPU-Only-007ACC?style=for-the-badge&logo=amd&logoColor=white" alt="CPU Only"> <img src="https://img.shields.io/badge/Machine%20Learning-FF6B6B?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning"><h3>My First Machine Learning Project: Plant Image Recognition with Incremental Learning</h3></div>

<h2>📋 About the Project</h2>
This is my first attempt at creating a comprehensive image recognition application, featuring incremental learning capabilities. The project addresses real-world machine learning challenges including catastrophic forgetting, weight balancing, and model optimization.
<br><br>
Important Design Decision: This project is specifically configured for CPU-only operation. The code intentionally disables GPU usage because I have an AMD GPU that isn't well-supported by TensorFlow. This makes the project accessible to everyone regardless of their hardware.
<br><br>
Development Journey: I spent significantly more time than expected, facing numerous challenges with catastrophic forgetting, weight balance issues, and poor learning rates. With assistance from DeepSeek AI, I implemented extensive diagnostics throughout all processes, allowing me to debug and improve the code iteratively.

## ✨ Features
<h3>🎯 Core Capabilities</h3>
<img width="732" height="372" alt="image" src="https://github.com/user-attachments/assets/4496692b-5332-4837-8f6c-c58278bcc2c8" />

<h3>📊 Advanced Diagnostics</h3>
Confusion matrices for both old and new data

- Class-by-class accuracy analysis

- Training history visualization

- Weight balancing monitoring

- Forgetting rate calculation

## 🚀 Quick Start
<h3>Prerequisites</h3>
<img width="201" height="176" alt="image" src="https://github.com/user-attachments/assets/9f488d61-7a93-49b6-906d-96d2949b5612" />

<h3>Installation</h3>

1. Clone the repository
git clone https://github.com/yourusername/plants-recognition.git
cd plants-recognition

2. Install dependencies
pip install -r requirements.txt

3. Prepare your data
<img width="412" height="175" alt="image" src="https://github.com/user-attachments/assets/87c5c91e-c911-43c6-963e-f20e23a6ea59" />

4. Run the application
python main.py

<h2>⚙️ Usage</h2>
<h3>Interactive Menu</h3>
<img width="372" height="218" alt="image" src="https://github.com/user-attachments/assets/ab862ed0-9895-4037-bc87-95123ecc7ac1" />

<h3>Command Examples</h3>

- Train a New Model
> t

Creates train/validation/test splits

Shows sample training images

Trains MobileNetV2 with augmentation

Displays accuracy and loss graphs

<img width="1788" height="837" alt="Снимок экрана 2025-12-27 213649" src="https://github.com/user-attachments/assets/7486f48d-5469-40cc-a9ae-20a302d410b4" />
<img width="1213" height="1126" alt="Снимок экрана 2025-12-27 213710" src="https://github.com/user-attachments/assets/f770dd13-9875-46bf-93cc-fecc3df948c6" />
<img width="1202" height="791" alt="Снимок экрана 2025-12-27 213734" src="https://github.com/user-attachments/assets/aa58e284-2ca5-4912-abd8-2d29f14d38c7" />
<img width="1496" height="566" alt="Снимок экрана 2025-12-27 132656" src="https://github.com/user-attachments/assets/bd6d21fd-fbcf-4e03-ba87-59ad74a9dd0b" />
<img width="1756" height="1346" alt="Снимок экрана 2025-12-27 083654" src="https://github.com/user-attachments/assets/c5746eb0-dbf1-46c1-99f3-05eb080cfc03" />

- Incremental Learning
bash
> i
> 
Creates automatic backups

Balances old/new class weights (1.5:0.7 ratio)

Freezes base layers, fine-tunes classifier

Includes old data to prevent forgetting

<img width="728" height="957" alt="Снимок экрана 2025-12-27 213831" src="https://github.com/user-attachments/assets/09201ee8-acd4-4ddc-bd86-81bd5455d11c" />

<img width="346" height="815" alt="Снимок экрана 2025-12-27 213853" src="https://github.com/user-attachments/assets/eae2ca2b-d673-484c-877e-e2aba4a630fe" />

<img width="250" height="560" alt="Снимок экрана 2025-12-27 213913" src="https://github.com/user-attachments/assets/8dc3b5d2-5425-4b95-875b-dbd38331ae31" />

<img width="1201" height="1126" alt="Снимок экрана 2025-12-27 213936" src="https://github.com/user-attachments/assets/4ee0edd5-d76e-45b8-b661-159c82f2d3c0" />

<img width="1198" height="650" alt="Снимок экрана 2025-12-27 214007" src="https://github.com/user-attachments/assets/f0da1c31-4bc4-4b52-973f-cfebdd9a7fcb" />

<img width="1497" height="1400" alt="Снимок экрана 2025-12-27 214816" src="https://github.com/user-attachments/assets/dc2c21b8-8fd5-4bca-918d-d4d883413660" />

<img width="941" height="1049" alt="Снимок экрана 2025-12-27 215816" src="https://github.com/user-attachments/assets/54331df5-f246-4fba-a282-70ade2d540af" />


- Recognize an Image
> r

Opens file dialog to select image

Processes image through model

Displays top-5 predictions with confidence

Provides confidence analysis

Example Output:

<img width="1808" height="1085" alt="image" src="https://github.com/user-attachments/assets/26eb4b0b-e4ab-42a6-8c7e-dc49a79028b3" />
<img width="1713" height="1066" alt="image" src="https://github.com/user-attachments/assets/a25ac706-b9c4-4d5f-a575-34373ec06bc4" />
<img width="1198" height="561" alt="image" src="https://github.com/user-attachments/assets/91a77337-b6b7-47a6-9938-2d904d92dba7" />
<img width="1855" height="1071" alt="image" src="https://github.com/user-attachments/assets/b862f8c4-0177-44a3-bdcb-8b4b9d6bf353" />
<img width="1195" height="562" alt="image" src="https://github.com/user-attachments/assets/0efb3730-6d14-4948-840e-d8676edaffa8" />

- Evaluate on examples
> ev

<img width="1316" height="877" alt="Снимок экрана 2025-12-27 215924" src="https://github.com/user-attachments/assets/c3be936a-cc1f-4098-be97-bea1b93dfcf8" />

<img width="397" height="447" alt="Снимок экрана 2025-12-27 220003" src="https://github.com/user-attachments/assets/15df4513-1d94-43e2-8902-eb9e03d719d8" />

<img width="397" height="877" alt="Снимок экрана 2025-12-27 220026" src="https://github.com/user-attachments/assets/91b35d0f-b18b-4adf-80d3-ffc545f5251d" />

<img width="378" height="398" alt="Снимок экрана 2025-12-27 220052" src="https://github.com/user-attachments/assets/74375b83-9cf7-4818-8b7e-4bd7ac560248" />

<img width="188" height="119" alt="Снимок экрана 2025-12-27 220124" src="https://github.com/user-attachments/assets/385092ca-d9da-41a2-ae2f-4a8e579589cf" />

<h2>🏗️ Architecture</h2>
<h3>Model Structure</h3>

<img width="430" height="311" alt="image" src="https://github.com/user-attachments/assets/3d60e157-467a-4033-943c-4ebf070a8f29" />

<h3>Data Pipeline</h3>

1. Image Loading → PIL Image conversion

2. Resizing → 224×224 (MobileNetV2 requirement)

3. Normalization → /255.0 scaling

4. Augmentation → Random flips, rotations, zoom, contrast

5. Batching → 32 images per batch

<h3>Technical Specifications</h3>

<img width="727" height="315" alt="image" src="https://github.com/user-attachments/assets/ede85f43-b1b5-46db-a371-95249f2a64d7" />

<h2>🧠 Challenges & Solutions</h2>
<h3>The Catastrophic Forgetting Problem</h3>
Problem: When training on new classes, the model completely forgets old classes.

My Solutions:

<img width="749" height="306" alt="image" src="https://github.com/user-attachments/assets/c052446f-ec66-44ab-ac10-05d1824f0614" />

<h3>CPU Optimization</h3>
Since I have an AMD GPU (not well-supported by TensorFlow), I configured the system for optimal CPU performance:

<h2>🔮 Project Roadmap</h2>
<h3>Completed</h3>

✅ Basic image recognition with MobileNetV2

✅ Incremental learning framework

✅ Extensive diagnostic tools

✅ CPU optimization for AMD hardware

✅ Backup system

<h3>In Progress</h3>

🔄 Better forgetting prevention strategies

🔄 Improved weight balancing algorithms

🔄 More efficient CPU training

<h3>Planned</h3>
📋 Mobile application

<h2>📁 Project Structure</h2>

<img width="498" height="297" alt="image" src="https://github.com/user-attachments/assets/64580166-72a4-46e5-816a-a5def1a780ac" />

<h2>🤝 Contributing</h2>
This is primarily a learning project, but contributions are welcome! Areas that need improvement:

Better catastrophic forgetting solutions (my biggest challenge)

More efficient CPU training algorithms

Additional data augmentation techniques

Performance optimization for larger datasets

<h2>💭 Final Thoughts</h2>
"This project represents my first serious dive into machine learning. I spent much more time than expected, facing challenges with catastrophic forgetting, weight balancing, and debugging. The most valuable lesson was understanding that incremental learning is much harder than it initially appears. Despite the difficulties, every error message and failed experiment taught me something new about how neural networks really work."

"The code intentionally uses CPU-only configuration because I have an AMD GPU. This actually made the project more accessible and forced me to optimize for efficiency. While the solution isn't perfect, the learning experience was invaluable."

<div align="center"><h2>Happy Learning and Coding! 🌱</h2></div>
