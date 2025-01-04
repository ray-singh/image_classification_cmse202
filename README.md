# Group 4: Breast Cancer Image Classification - CMSE202 (Section 4) Final Project
### Members: Rayansh Singh, Saikeertana Gubbala, Calvin DeJong, Chris Rogers

## Abstract

Invasive Ductal Carcinoma (IDC) is the most common subtype of breast cancer, accounting for approximately 80% of diagnosed cases. Early detection is crucial for effective treatment because IDC has the potential to metastasize, or spread, to other parts of the body. Presently, IDC detection relies heavily on the manual analysis of tissue slices by pathologists which is a process that is not only time-consuming but also prone to human error and inconsistencies. Furthermore, the manual method may be impractical or inaccessible in regions lacking medical expertise and advanced diagnostic resources. This project aims to overcome these challenges by leveraging deep learning techniques to automate the detection of IDC in breast tissue samples.

Our primary research question investigates whether a neural network-based approach can accurately classify image patches of breast tissue as IDC-positive or IDC-negative. To address this, we designed and trained two types of neural network models: a Feedforward Neural Network (FNN) and a Convolutional Neural Network (CNN). We used a substantial and imbalanced dataset from Kaggle, containing 277,524 image patches, each measuring 50x50 pixels. The dataset, drawn from 162 patients, is inherently skewed, with a higher proportion of IDC-negative samples (198,738) compared to IDC-positive samples (78,786). This class imbalance posed an additional challenge, which we addressed through techniques such as using class weights during model training.

To develop a robust neural network for IDC classification, we used PyTorch, a powerful Python module tailored for deep learning applications. The FNN we implemented consists of five layers of nodes, each connected by ReLU (Rectified Linear Unit) activation functions, designed to handle the flattened 7,500-dimensional input vector derived from the image patches. Additionally, we explored a CNN architecture with three convolutional layers, each followed by max-pooling layers, to learn spatial features from the images, and concluded with three fully connected feedforward layers.

After training the models, we tested them on unseen data to evaluate their performance. The FNN achieved an accuracy of approximately 80%, while the CNN performed less effectively, with an accuracy of around 65%. To gain a comprehensive understanding of the FNN's predictive capabilities, we calculated various performance metrics, including precision, recall, and the ROC (Receiver Operating Characteristic) curve. We also visualized the model’s results using Matplotlib, plotting images with their true and predicted labels to assess classification outcomes.

Our findings demonstrate that automating IDC detection with deep learning models is feasible, with the FNN yielding promising results. Importantly, our final FNN model is compact (<50MB), making it efficient for deployment even on older or resource-constrained systems/computers. We see this model serving as a valuable tool to assist licensed physicians in detecting IDC, especially in settings where rapid and reliable diagnostic support is necessary.

## How to Run the Project

This project focuses on classifying breast cancer image patches using a convolutional neural network (CNN) built with PyTorch. The instructions below outline how to set up and run the code.

### Prerequisites

You will need to install the following Python libraries:
- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `matplotlib`

You can install all the required packages using `pip`:

```bash
pip install torch torchvision numpy pandas matplotlib
```
## Steps for running: 
### 1. Download the dataset
   * Use the link (https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).
   * Extract the contents of the zip file and ensure the folder structure matches the paths specified in the code (e.g., '/path_to/cancer_images/IDC_regular_ps50_idx5/**/*.png').
### 2. Train the model
* Open trainmodel.ipynb to train the feedforward neural network on the image data.
* Update the file path to point to the location of your extracted dataset.
* The model weights will be saved as ```model_weights.pth```. Note that training can take approximately 30 minutes, so you may skip this step and use the pre-trained model provided.
### 3. Visualize the results 
* Navigate to the "model eval - ray" folder.
* Run ```simple_nn.ipynb``` to see the visualizations of the model's performance on the test data.

* Note: Step 2 is optional. If you prefer not to train the model yourself, use the provided ```model_weights.pth``` file for evaluation.


## Contributions from Group Members
### Calvin DeJong: 
I contributed to dataset curation and developed code for both pre-trained and custom-trained models. I explored the functioning of feedforward and convolutional neural networks, implementing these on our dataset. Additionally, I assisted teammates in troubleshooting compatibility issues, providing solutions and code to help navigate those problems. I also prepared and contributed to presentation slides, explaining the model's functionality and the underlying mathematics. Lastly, I collaborated on writing the abstract, ensuring a cohesive summary of our project. 

### Saikeertana Gubbala
I worked a bit on dataset curation in the first class, and did some research on what sorts of modules and libraries might work well with a dataset of our size. After Calvin told us what might work the best (PyTorch), I looked into how that can be implemented and had some initial code written that we could look over as a group. I dealt more with group organization, making sure to send zoom links, discuss our project in detail, ensure everyone had access to files, GitHub repository, and presentations. I also worked on the presentation(s), abstract, README files as well. Also, worked in a group to troubleshoot any issues that came up with the code. Specifically, I focused on the background, research questions, context and models of our overall project. 

### Rayansh Singh:

I researched projects based on similar datasets to understand which models, libraries, and techniques were best suited to our project. I was active in our group chat and meetings, sharing my insights and progress. I contributed towards implementing our convolutional neural network. After Calvin and Keertana succesfully implemented a feed-forward neural network, I utilized evaluation metrics such as confusion matrices, accuracy score, etc to assess the model’s effectiveness. I visualized key insights about the model's performance using Matplotlib, ensuring our findings were effectively communicated. I also worked on our presentation, writing about the challenges we faced and how we overcame them.


### Christopher Rogers:

I contributed by preparing presentation slides to effectively communicate our findings and methods. I researched alternative approaches and methods that could have been implemented to improve the accuracy of our results, given more time. Additionally, I played an active role in troubleshooting the code, identifying and resolving bugs to ensure smooth functionality. My efforts helped enhance both the technical and presentation aspects of the project, contributing to its overall success.
