# DSAI-course
## Setting Up Your Virtual Environment  

A virtual environment helps isolate dependencies and manage Python packages efficiently. Follow the steps below based on your operating system:  

### 1. Install Python and pip  
Ensure you have Python installed on your system. If you haven't installed it yet:  

- **Windows & macOS:** Download and install Python from [python.org](https://www.python.org/downloads/).  
- **Linux (Debian-based):**  
```bash
sudo apt-get install python3-pip
```

After installation, verify python and pip versions
```bash
python --version
pip --version
```

### 2. Create a virtual environment

Once Python and pip are installed, install `virtualenv` by running:
```bash
pip install virtualenv
```

Confirm the installation 
```bash
virtualenv --version
```

Choose a name for your environment (e.g., `DSAI_env`) and run:
```bash
virtualenv DSAI_env
```

For a specific Python version:
```bash
virtualenv -p python3 DSAI_env
```

There are different ways to activate virtual environment;

- Windows (Command Prompt):
```bash
DSAI_env\Scripts\activate
```

- Windows (PowerShell):
```bash
DSAI_env\Scripts\Activate.ps1
```

- macOS & Linux:
```bash
source DSAI_env/bin/activate
```

Once activated, you will see the environment name in your terminal prompt.

To exit the virtual environment, run:
```bash
deactivate
```

Now you're ready to install packages and work within an isolated environment for your Data Science and AI projects! 🚀

## Module 1 - DSAI 100:Introduction to data science
Day 1: Foundations of Data Science

- Introduction to data science concepts, terminology, and applications.
- Understanding data types, data sources, and data formats.
- Data collection methods and data ethics.

Day 2, 3: Data Acquisition and Preprocessing

- Techniques for data acquisition from various sources such as APIs, databases, and web scraping.
- Data cleaning and preprocessing methods for handling missing values, outliers, and inconsistencies.
- Introduction to data wrangling tools like pandas-profiling and data augmentation techniques.

Day 4, 5: Exploratory Data Analysis (EDA)

- Advanced data visualization techniques using libraries like Plotly, Bokeh, and Altair.
- Statistical methods for exploratory data analysis, hypothesis testing, and feature importance.
- Time series analysis and spatial data visualization.

Day 6, 7, 8: Statistical Modeling and Inference

- Introduction to statistical modeling techniques such as linear regression, logistic regression, and time series forecasting.
- Model evaluation metrics and validation techniques.
- Bayesian statistics and probabilistic programming with libraries like PyMC3.

Day 9, 10: Applied Machine Learning

- Overview of supervised and unsupervised learning algorithms.
- Ensemble methods: bagging, boosting, and stacking.
- Introduction to dimensionality reduction techniques such as PCA and t-SNE.

Hands-on project: Applying machine learning algorithms to real-world datasets, from data preprocessing to model evaluation

## Module 2: DSAI 101:Advanced Data Science Techniques

Day 1: Advanced Feature Engineering

- Feature engineering for structured and unstructured data, including text and image data.
- Feature extraction from time series data and geospatial data.
- Automated feature engineering with libraries like FeatureTools and auto-sklearn.

Day 2, 3: Model Optimization and Hyperparameter Tuning

- Techniques for model selection, hyperparameter tuning, and model interpretation.Curricula Modules Topics
-Grid search, random search, and Bayesian optimization for hyperparameter tuning.
- Model interpretability techniques such as SHAP values and LIME.

Day 4, 5: Time Series Analysis and Forecasting

- Advanced time series analysis techniques, including ARIMA, SARIMA, and Prophet.
- Forecasting methods for hierarchical time series and multivariate time series.
- Anomaly detection and intervention analysis in time series data.

Day 6, 7, 8: Unsupervised Learning and Clustering

- Clustering algorithms: k-means, hierarchical clustering, DBSCAN.
- Dimensionality reduction techniques: t-SNE, UMAP, autoencoders.
- Applications of unsupervised learning in anomaly detection, customer segmentation, and recommender systems.

Day 9, 10: Big Data and Scalable Machine Learning

- Introduction to distributed computing frameworks: Apache Spark and Dask.
- Scalable machine learning algorithms for big data analysis.

Hands-on project: Implementing scalable machine learning pipelines on large-scale datasets using distributed computing

## Module 3: DSAI 102:Artificial Intelligence and Deep Learning frameworks.

Day 1: Introduction to Artificial Intelligence and Neural Networks

- Fundamentals of artificial intelligence, neural networks, and deep learning.
- Activation functions, loss functions, and optimization algorithms.
- Introduction to deep learning frameworks: TensorFlow and PyTorch.

Day 2, 3: Convolutional Neural Networks (CNNs)

- Understanding CNN architecture, convolutional layers, pooling layers, and fully connected layers.
- Transfer learning and fine-tuning pre-trained CNN models.
- Advanced CNN architectures: ResNet, Inception, and DenseNet.

Day 4, 5: Recurrent Neural Networks (RNNs) and Sequence Models

- Introduction to RNN architecture, long short-term memory(LSTM), and gated recurrent units (GRU).
- Sequence-to-sequence models for machine translation and speech recognition.
- Attention mechanisms and transformer architectures.

Day 6, 7, 8: Generative Models and Reinforcement Learning

- Introduction to generative models: variational autoencoders
- Reinforcement learning fundamentals: Markov decision processes, policy gradients, and value iteration.
- Deep reinforcement learning algorithms: Deep Q-Networks (DQN), policy gradients (PG), and actor-critic methods.

Day 9, 10: Advanced Deep Learning Applications and Project

- Applications of deep learning in computer vision, natural language processing, and reinforcement learning.
- Cutting-edge research topics: self-supervised learning, meta-learning, and multi-modal learning.

Capstone project: Students choose a project related to their interests or career goals, applying advanced deep learning techniques to solve a specific problem or explore a research question. They present their project outcomes and findings to the class.
(VAEs) and generative adversarial networks (GANs).
