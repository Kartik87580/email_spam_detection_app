# 📧 Email Spam Classifier using NLP

A simple and interactive **Email Spam Detection** web app built with **Streamlit**. The app uses **Natural Language Processing (NLP)** and a **Machine Learning model** to classify email messages as **Spam** or **Not Spam** in real time.

---

## 🚀 Features

* Enter any email or message content
* Instantly get spam classification result
* Uses a trained model with **TF-IDF** and **Multinomial Naive Bayes**
* Clean and simple web interface with Streamlit

---

## 🧠 Tech Stack

* **Python**
* **Streamlit** – web interface
* **Scikit-learn** – machine learning
* **Pickle** – to load model and vectorizer
* **TF-IDF Vectorizer** – feature extraction for text
* **Naive Bayes Classifier** – trained model

---

## 📂 Project Structure

```
email_spam_classifier/
│
├── app.py                  # Streamlit app file
├── email_spam_detection.ipynb  # Model training and preprocessing notebook
├── spam.csv                # Dataset of email messages
├── spam_model.pkl          # Trained classifier
├── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer
└── README.md               # Project documentation
```

---

## 📥 Dataset

* The dataset used: **spam.csv**
* It contains email messages labeled as **spam** or **ham (not spam)**

---

## 📈 How It Works

1. The app takes user input (email text)
2. It vectorizes the input using a pre-trained **TF-IDF vectorizer**
3. Then it passes the vector to a trained **Multinomial Naive Bayes** model
4. Returns prediction as **SPAM** or **NOT SPAM**

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/email_spam_classifier.git
   cd email_spam_classifier
   ```

2. Install required packages:

   ```bash
   pip install streamlit scikit-learn pandas
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---



## ✅ Example

* **Input**:

  ```
  Congratulations! You've won a $1000 Walmart gift card. Click here to claim.
  ```

* **Output**:
  ❗ This email is SPAM

---

## 🧪 Model Details

* **Text Vectorizer**: TF-IDF
* **Classifier**: Multinomial Naive Bayes
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
* Trained using `email_spam_detection.ipynb`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---


