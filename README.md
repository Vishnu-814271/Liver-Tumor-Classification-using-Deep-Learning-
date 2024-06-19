# Liver-Tumor-Classification-using-Deep-Learning-
**Title: Liver Tumour Classification**

Keywords: - Machine Learning in Medicine, Medical Inventory 
Management, Healthcare Supply Chain, Predictive Analysis, Hospital Management

**1. Introduction:**
The adoption of AI in the medical industry is significantly enhancing diagnostic practices[03], especially in the intricate area of liver tumor identification. This project, in collaboration with a prominent South Indian hospital, seeks to transform the traditional approach of relying on manual image interpretation by clinicians into one that utilises AI for more accurate and quicker diagnostics. However, manual liver pathology examination requires considerable time and labour[04].
The liver, a critical organ second only in size to the skin, typically weighs between 1.2 and 1.5 kilograms and is divided into two main lobes[16]. It lies adjacent to the gallbladder, pancreas, and intestines, functioning primarily through hepatocytes, which account for 70-85% of its mass. These cells are essential for various functions including metabolism and detoxification. Liver tumors can range from benign to malignant, with Hepatocellular Carcinoma (HCC) being the most prevalent type of primary liver cancer[01], frequently associated with chronic liver diseases like cirrhosis. Another notable but rarer malignancy is Intrahepatic Cholangiocarcinoma, which forms within the bile ducts of the liver.[16]
By applying deep learning technology, our project categorises liver imaging into three classifications: Cholangiocarcinoma, Hepatocellular Carcinoma (HCC), and Normal Liver. This methodology not only expedites the diagnostic process but also increases its precision, thus significantly improving over traditional methods.
Our project aims to markedly enhance the accuracy of liver tumor diagnostics, reduce the costs associated with these diagnoses, and improve patient outcomes. Our objectives include raising the liver tumor diagnosis rate by at least 10%, achieving a classification accuracy of at least 96%, and attaining substantial savings in diagnostic expenses. These advancements are expected to benefit patient care and provide significant economic benefits by optimizing resource utilization in healthcare settings.

**1. DenseNet121:**
•	Developed by the Visual Geometry Group at the University of Oxford.
•	Known for its simplicity and uniformity.
•	Consists of 16 layers with 13 convolutional layers (3x3 filters) and 3 fully connected layers.
•	Achieved remarkable performance in the 2014 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) with 92.7% classification accuracy.
•	Captures fine-grained details due to small filter sizes.

**2.	ResNet50:**
•	Introduced by Microsoft Research in 2015.
•	Features 50 layers and innovative residual blocks.
•	Residual connections mitigate the vanishing gradient problem.
•	Achieves 92.2% top-5 classification accuracy on the ImageNet benchmark dataset.

**3.	VGG16:**
•	Developed by the Visual Geometry Group at the University of Oxford.
•	Consists of 16 layers (13 convolutional and 3 fully connected).
•	Achieved remarkable performance in the 2014 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) with 92.7% classification accuracy.
•	Known for its simplicity and uniformity.
•	Captures fine-grained details due to small filter sizes.
 
**2. Methodology and Techniques**
**2.1 Data Collectio**n
Histopathological images and tumor information were gathered from the GDC-Portal (https://portal.gdc.cancer.gov/). 210 histopathological images of liver tumors (70 HCC liver images, 70 normal liver images, and 70 cholangiocarcinoma images) are presented in this paper. They are 856 x 856 pixels. These images were available in two different ways: open access and control access. We are collecting open-access datasets.[18]
**2.2 Data Preprocessing**
Medical image analysis plays a pivotal role in diagnosing diseases, monitoring treatment progress, and aiding clinical decision-making. However, before feeding these images into deep learning models, a robust preprocessing pipeline is essential to enhance data quality, mitigate noise, and ensure optimal model performance.
 
**1. Image Format and Metadata**
DICOM Format: Most medical images are stored in the Digital Imaging and Communication in Medicine (DICOM) format. Unlike standard image formats (e.g., PNG or JPEG), DICOM files include rich metadata such as patient information, acquisition parameters, and windowing intervals. This comprehensive data consolidation facilitates seamless communication between devices and ensures data integrity.
**2. Hounsfield Unit Transformation**
HU Conversion: Transforming raw pixel values to Hounsfield Units (HU) is a crucial step. HU quantifies the radiodensity of tissues, allowing us to differentiate between various anatomical structures. For instance, air has a HU value of -1000, while bone ranges from 100 to 1000.
****3. Noise Reduction**
**Noise Filtering: Medical images often suffer from noise due to acquisition imperfections or artefacts. Applying filters (e.g., Gaussian or median filters) can suppress noise while preserving relevant features.
**4. Tilt Correction**
   Image Alignment: Correcting any tilt or misalignment in the images ensures consistent orientation across the dataset. This step is particularly crucial for volumetric scans.
**5. Cropping and Padding**
Region of Interest (ROI): Cropping the image to focus on the relevant anatomical region (e.g., liver or tumor) reduces computational load and enhances model efficiency.
Padding: Adding padding to ensure uniform image dimensions is essential for batch processing within neural networks.
**6. Normalization and Standardization**
Pixel Intensity Normalization: Rescaling pixel intensities to a common range (e.g., [0, 1]) ensures consistent input for the model.
Z-score Standardization: Standardizing pixel values using mean and standard deviation helps stabilize training and improves convergence.
**7. Data Augmentation**
Augmenting Variability: Introducing variations (e.g., rotations, flips, or zooms) artificially expands the dataset, reducing overfitting and enhancing model generalization.
8. Quality Control and Annotation
Manual Inspection: Regularly inspect images for artefacts, misalignments, or inconsistencies.
Expert Annotations: Annotating regions of interest (e.g., tumour boundaries) by experts ensures accurate ground truth labels.
 
This study provides an overview of the data and deep learning framework proposed. The images analyzed were obtained from the TCGA database (left panel). To train and test our models, we divided images from the TCGA dataset into a training set (80%) and a test set (20%). We used samples from TMAs in the WCH biobank for external validation. Each slide or TMA dot was tiled into non-overlapping 256×256 pixel patches, with tiles containing more than 12.5% background being excluded after RGB normalization . Tiles extracted from training sets served as inputs to both tasks one and two in our models; these performances tested on tilesets accessed under the testing section while also providing insights about limitations through task conduction over multiple repeated instances located at various coordinates using TMAs sampled during validations exercises by challenge trained model algorithms robustness[20]

**2.3 Model Approach**
We tried out a bunch of models: DenseNet121, ResNet50, ResNet150, ResNet151, and VGG16. We were looking at their training and testing accuracies, and we set a goal for ourselves - we wanted to hit at least 96% accuracy.
DenseNet121 demonstrated superior performance, achieving a training accuracy of 98.21% and a testing accuracy of 98%, exceeding the set benchmark. The architecture of DenseNet121 promotes feature reuse and mitigates the vanishing-gradient problem, enhancing the model’s learning efficiency and generalization capability.[22]
In comparison, ResNet50 achieved a testing accuracy of 100%, which may indicate overfitting. Both ResNet150 and ResNet151 did not meet the benchmark accuracy, suggesting a need for further optimization. VGG16 had the lowest testing accuracy of 72%, indicating its limitations for this specific task.[21]
The distinguishing factor of DenseNet121 is its dense connectivity pattern. Unlike ResNets or VGG16, where each layer is only connected to the next one, DenseNet121 connects each layer to every other layer. This dense connectivity facilitates better information flow and gradient propagation, leading to improved learning outcomes.
DenseNet121: Exhibiting a Train Accuracy of 98.21% and Test Accuracy of 98%, DenseNet121 not only surpassed the required benchmark but also demonstrated robustness against overfitting. The architecture’s design promotes feature reuse and reduces the vanishing-gradient problem, which enhances learning efficiency and generalization.[22]
Comparison with Other Models:
o ResNet50 achieved a Test Accuracy of 100%, suggesting potential overfitting, which necessitates further investigation.[21]
o ResNet150 and ResNet151 fell short of the required accuracy, indicating a need for model optimization or data augmentation strategies.[21]
o VGG16 underperformed significantly, with a Test Accuracy of 72%, highlighting the limitations of older architectures in handling complex image classification tasks.[21]
Dense Net vs. Other Models: DenseNet121’s dense connectivity pattern, where each layer receives input from all preceding layers, contrasts with the residual connections in ResNets and the sequential layering in VGG16. This unique feature fosters better gradient flow and feature propagation, leading to improved learning outcomes.
Superiority of DenseNet121: The superior performance of DenseNet121 can be attributed to its architectural advantages, which include:
o Enhanced feature propagation and reuse,
o Effective parameter utilization,
o Improved gradient flow, and
o Strong generalization capabilities.
 
**In conclusion**, DenseNet121 was the clear winner in our study. It’s not just efficient, but also effective for our image classification tasks. And the best part? It’s robust against overfitting, as shown by its consistent accuracy across the training and testing phases. So, if you’re looking to classify liver tumours (or anything else, really), DenseNet121 is definitely worth considering![22]
2.4 Research Findings
The primary objective of this research was to ascertain the efficacy of various deep learning models in the classification of liver tumours using image data. Among the models tested, DenseNet121 was found to be particularly effective, as evidenced by its high accuracy rates on both training (98.21%) and testing datasets (98%).[22]
Model name	Train Accuracy	Test Accuracy	Required Accuracy
Densenet121	98.21%	98%	96%
ResNet50	99.40%	100%	96%
ResNet150	94.00%	92%	96%
ResNet151	88.10%	88%	96%
VGG16	58.49%	72%	96%
			
