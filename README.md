Brain Tumor Classification with MRI Images
Overview
This project implements a deep learning model to classify brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. Using the ResNet101 architecture with transfer learning, the model achieves a test accuracy of 97.4% on the Brain Tumor MRI Dataset. The project demonstrates advanced image processing, model fine-tuning, and performance visualization for medical imaging applications.
Features

Data Preprocessing: Resized MRI images to 224x224, converted to RGB, and normalized pixel values for model input.
Exploratory Data Analysis: Analyzed class distributions across 5712 training and 1311 testing images.
Model Architecture: Utilized ResNet101 (pre-trained on ImageNet) with fine-tuned layers, GlobalAveragePooling2D, and dense layers with dropout for classification.
Training: Trained the model for 30 epochs with class weights to handle class imbalance and dynamic learning rate scheduling.
Evaluation: Achieved 97.4% accuracy on the test set, with confidence score visualizations for correct and misclassified samples.
Visualization: Generated scatter plots of confidence scores using Matplotlib to analyze model predictions.

Technologies Used

Python: Core programming language.
TensorFlow/Keras: For building and training the ResNet101 model.
NumPy & PIL: For image preprocessing and data manipulation.
Matplotlib: For visualizing confidence scores.
Scikit-learn: For computing class weights.
OpenCV: For image processing utilities.

Dataset

Brain Tumor MRI Dataset (sourced from Kaggle).
Training Set: 5712 images across four classes (glioma, meningioma, no tumor, pituitary).
Testing Set: 1311 images for evaluation.
Images are organized in folders by class, accessible via /kaggle/input/brain-tumor-mri-dataset/.

Installation

Clone the repository:git clone https://github.com/Aerglo/brain-tumor-classification.git


Install dependencies:pip install tensorflow numpy matplotlib pillow scikit-learn opencv-python


Download the Brain Tumor MRI Dataset and place it in the data/ folder.
Run the Jupyter notebook:jupyter notebook brain_tumor_classification.ipynb



Usage

Open the brain_tumor_classification.ipynb notebook.
Execute cells to preprocess data, train the ResNet101 model, and visualize results.
Outputs include:
Model accuracy metrics (97.4% test accuracy).
Scatter plots showing confidence scores for correct and misclassified predictions.



Results

Test Accuracy: 97.4% after 30 epochs, with 99.9% training accuracy.
Key Insights: The model effectively distinguishes between tumor types, with minimal misclassifications visualized via confidence score plots.
Class Balance: Class weights ensured robust performance despite potential dataset imbalances.

Future Improvements

Experiment with other architectures (e.g., EfficientNet, VGG16) for comparison.
Incorporate data augmentation (e.g., rotation, flipping) to improve generalization.
Add confusion matrix and ROC curves for detailed performance analysis.
Explore deployment as a web-based diagnostic tool.

Contact
For questions or feedback, reach out to me at imnima82@gmail.com or via GitHub.
