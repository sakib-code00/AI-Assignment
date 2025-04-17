<h1>Spam Email Detection</h1>
<section id="sec1">
  <h2>Project Overview</h2>
  <p>This project focuses on building a machine learning model to detect spam emails. Using a labeled dataset of emails (spam and ham), the model learns to identify unwanted or harmful messages. The process includes data cleaning, feature extraction, model training (KNN, Naive Bayes, Logistic Regression, etc.), and evaluation. The final model can predict whether a new email is spam or not based on its content.</p>
</section>
<h2>Table of Content</h2>
<ul>
  <li><a href="#sec1">Project Overview</a></li>
  <li><a href="#sec2">Data Description</a></li>
  <li><a href="#sec3">Data Processing</a></li>
  <li><a href="#sec4">Exploratory Data Analysis</a></li>
  <li><a href="#sec6">Model Selection</a></li>
  <li><a href="#sec7">Training and Evaluation</a></li>
  <li><a href="#sec9">Future work</a></li>
  <li><a href="#sec10">Conclusion</a></li>
  <li><a href="#sec11">Visualization</a></li>
  <li><a href="#sec12">Data Source</a></li>
    
</ul>
<section id="sec2">
  <h2>Data Descriptions</h2>
  <p>The dataset contains a collection of labeled emails used for spam detection. Each entry includes:</p>
  <ul>
    <li><b>Message:</b> The actual content of the email as plain text.</li>
    <li><b>Label:</b> Indicates whether the email is spam or ham (not spam).</li>
  </ul>
  <p>The dataset is used to train and test the model to classify emails based on their text content.</p>
</section>
<section id="sec3">
  <h2>Data Processing</h2>
  <p>For the data to be ready for the machine learning model, we performed the following steps:</p>
  <ol type="a">
    <li>Drop Null Values – Removed any rows with missing or null values.</li>
    <li>Delete Duplicates – Removed any duplicate email entries to avoid bias.</li>
    <li>Converted text into numerical values to enable machine learning algorithms to process them.</li>
  </ol>
  <p>These steps help ensure clean and consistent data for training the model.</p>
</section>
<section id="sec4">
  <h2>Exploratory Data Analysis</h2>
  <p>In this step, we explore the dataset to understand its structure, identify patterns, and detect any issues that need to be addressed. The key steps of EDA are:</p>
  <ul>
    <li>Display basic information about the dataset, such as the number of rows, columns, and data types.</li>
    <li>Visualize the count of spam and ham emails to check for class imbalance.</li>
    <li>Inspect if there are any null or missing values in the dataset.</li>
  </ul>
  <p>By conducting EDA, we gain insights into the data, helping to make informed decisions for preprocessing and model building.</p>
</section>
<section id="sec6">
  <h2>Model Selection</h2>
  <p>For this project, we tested and compared the following machine learning models:</p>
  <ul>
    <li>
      <b>Logistic Regression:</b>
      <ul>
        <li>A reliable and interpretable model for binary classification.</li>
        <li>Performs well with text features like TF-IDF</li>
      </ul>
      <li>
      <b>K-Nearest Neighbors (KNN):</b>
      <ul>
        <li>Classifies emails based on similarity to nearby data points.</li>
        <li>Slower with large datasets but simple to understand.</li>
      </ul>
    </li>
    <li>
      <b>Naive Bayes:</b>
      <ul>
        <li>Works well for text classification problems.</li>
        <li>Fast and accurate for spam detection.</li>
      </ul>
    </li>
    </li>
  </ul>
</section>

<section id="sec7">
  <h2>Training and Evaluation</h2>
  <p>Split the data into training and validation sets to evaluate the models’ effectiveness. The following metrics were used:</p>
  <li>Accuracy: Measures the overall correctness of the model.</li>
  <li>ROC-AUC Score: Evaluates model performance, especially useful for imbalanced datasets.</li>
  <li>Confusion Matrix: Helps visualize the classification results and identify misclassifications.</li>
  <li>Classification Report: Provides detailed metrics like precision, recall, and F1-score for each class.</li>
  <h3>Model Training Results</h3>
  <p>The performance of each model is summarized below:</p>
  <table style="border: 1px solid black;">
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.98</td>
    </tr>
    <tr>
      <td>KNN</td>
      <td>0.91</td>
    </tr>
    <tr>
      <td>Naive Bayes</td>
      <td>0.99</td>
    </tr>
  </table>
  <p>The Naive Bayes model achieved the highest validation accuracy making it the best-performing model.</p>
</section>

<section id="sec9">
  <h2>Future Work</h2>
  <p>To improve the performance and usability of this project in the future, the following steps can be considered:</p>
  <ol>
    <li><b>Real-Time Spam Detection:</b> Develop a system that can detect spam emails in real-time.</li>
    <li><b>Larger & Diverse Dataset:</b> Train the model on a larger and more diverse email dataset to improve generalization.</li>
    <li><b>Multilingual Support:</b> Extend the model to detect spam in different languages.</li>
    <li><b>Deploy as a Web App or API:</b> Make the model accessible through a user-friendly interface or REST API.</li>
  </ol>
</section>

<section id="sec10">
  <h2>Conslusion</h2>
  <p>In this project, we developed a machine learning model to detect spam emails accurately. After preprocessing the data and extracting features using CountVectorizer, we trained and evaluated several models including Naive Bayes, KNN, and Logistic Regression.</p>
  <p>Among them, Logistic Regression performed the best based on accuracy, precision, recall, and F1-score. The model was able to effectively distinguish between spam and ham emails.</p>
  <p>This project shows that machine learning can be a powerful tool in building automated spam filters, helping users stay safe and reducing unwanted emails.</p>
</section>

<section id="sec11">
  <h2>Visualization</h2>
  <h3>Columns</h3>
  <img src="https://github.com/user-attachments/assets/075e9e88-1200-490f-a29a-c36e923a8239">

  <h3>Precipitation level in Sylhet in 2023</h3>
  <img src="https://github.com/user-attachments/assets/6d0f99fe-3d4b-463c-a7e3-0fb92458e493">
  <h3> The distribution of the continuous features given in the dataset</h3>
  <img src="https://github.com/user-attachments/assets/e939ff5d-944c-4d88-9c2b-1dab9b05a97c" alt="Subplots">
  <h3> Boxplots for the continuous variable to detect the outliers present in the data.</h3>
  <img src="https://github.com/user-attachments/assets/791ce6b2-4d44-48b6-bc8e-ab4e4d30b646" alt="Boxplots">
  <h3>Correlation Heatmap</h3>
  <img src="https://github.com/user-attachments/assets/90b5da86-780a-475b-b1b4-b8342d686949" alt="Heatmap">
  <h3> confusion matrix as well for the validation data using the Logistic Regression model</h3>
  <img src="https://github.com/user-attachments/assets/106d623c-f9ea-42ab-b04e-076ae1d7588a)" alt="Logistic Regression ">
  <h3> confusion matrix as well for the validation data using the XGBoost model</h3>
  <img src="https://github.com/user-attachments/assets/beb6b4e8-c37f-4f54-b7f7-4ecbf1272c17" alt="XGBoost">
  <h3> confusion matrix as well for the validation data using the SVC (Support Vector) model</h3>
  <img src="https://github.com/user-attachments/assets/3e00ffc2-2ca4-402c-be89-b92bfcce0a18" alt="SVC">
  <h3> confusion matrix as well for the validation data using the Random Forest model</h3>
  <img src="https://github.com/user-attachments/assets/0c223560-e067-40a1-af05-8cadbb32d8a7" alt="Random Forest">
  <h3> confusion matrix as well for the validation data using the K-Nearest Neighbors model</h3>
  <img src="https://github.com/user-attachments/assets/2788b1f8-71c5-4ae3-b595-5102d4482394" alt="K-NN">
  <h3>Decision Tree</h3>
  <img src="https://github.com/user-attachments/assets/cefb1519-ccbd-4f2a-a4fb-b574df53a8bb" alt="Decision Tree">
  <h3> confusion matrix as well for the validation data using the Decision Tree model</h3>
  <img src="https://github.com/user-attachments/assets/f6342213-cb1e-47ac-890c-79a215d936b2" alt="Decision Tree">
</section>

<section id="sec12">
  <h2>Data Source</h2>
  <h3><a href="https://drive.google.com/drive/folders/1jo152t3fAcijJ6o7fowG1iKq8Fob9Y61?usp=drive_link">Email Dataset</a></h3>
  
</section>
