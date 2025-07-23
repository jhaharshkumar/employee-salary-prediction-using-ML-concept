

# Employee Salary Classification App



This project is a web application that predicts whether an employee's annual salary is greater than $50,000 or less than or equal to $50,000. The prediction is based on demographic and employment-related features.

The application is built using Streamlit and powered by a Gradient Boosting Classifier model trained on the UCI Adult Census Income dataset.

## ✨ Features

-   **Interactive Sidebar:** Input employee details using sliders and dropdowns.
-   **Real-time Prediction:** Get an instant salary class prediction for a single employee.
-   **Batch Prediction:** Upload a CSV file with multiple employee records to get predictions for all of them at once.
-   **Downloadable Results:** Download the batch predictions as a new CSV file.

## 🛠️ Technology Stack

-   **Python 3.x**
-   **Streamlit:** For creating the interactive web application.
-   **Pandas:** For data manipulation and preprocessing.
-   **Scikit-learn:** For building and training the machine learning model.
-   **Joblib:** For saving and loading the trained model.

## 📂 Project Structure

```
.
├── app.py              # The main Streamlit web application
├── train_model.py      # Script to preprocess data and train the model
├── best_model.pkl      # The saved, trained Gradient Boosting model
├── adult.csv           # The raw dataset
├── requirements.txt    # Python package dependencies
└── README.md           # You are here!
```

## ⚙️ Setup and Installation

Follow these steps to set up the project on your local machine.

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/employee-salary-classification.git
cd employee-salary-classification
```

**2. Create and activate a virtual environment (recommended):**

-   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

**3. Install the required packages:**

A `requirements.txt` file is provided to install all necessary dependencies.

```bash
pip install -r requirements.txt
```

*(You will need to create this `requirements.txt` file)*:

```txt
# requirements.txt

streamlit
pandas
scikit-learn
joblib
```

## 🚀 How to Run

There are two main steps: training the model and then running the Streamlit application.

**Step 1: Train the Model**

Before you can run the app, you need to train the model and generate the `best_model.pkl` file. Run the training script from your terminal:

```bash
python train_model.py
```

This script will load `adult.csv`, preprocess the data, train the Gradient Boosting Classifier, and save the final model as `best_model.pkl`.

**Step 2: Run the Streamlit App**

Once the model is saved, you can launch the web application:

```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the running application.

## 🤖 The Model

-   **Algorithm:** The final model is a **Gradient Boosting Classifier**, which was selected for its high accuracy (around 85.7%) on the test set.
-   **Dataset:** The model was trained on the [UCI Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult).
-   **Preprocessing:** The following steps were taken to prepare the data for training:
    -   Handled missing values (represented as `?`).
    -   Removed outliers from `age` and `educational-num` columns.
    -   Dropped the redundant `education` column.
    -   Applied **Label Encoding** to convert all categorical features into numerical format.

