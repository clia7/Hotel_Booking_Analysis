# Hotel Booking Demand - Predicting Cancellations 🏨

This project is focusing on predicting passenger booking cancellations using machine learning. The entire pipeline was built, processed, and evaluated within a single end-to-end framework.
Using this clean, engineered data pipeline and a Random Forest classification model, the system identifies whether a reservation will be canceled (`is_canceled`), providing actionable insights for hotel revenue management.

📊 Project Overview & Results

* **Local Validation Score:** 87.99% Accuracy / Classification Report (Baseline evaluation via `train_test_split` with `random_state=42` and `stratify=y`)
* **Model Used:** Random Forest Classifier (`RandomForestClassifier` with `random_state=42`)
* **Feature Set:** Optimized features including engineered variables for total stays, revenue proxies, guest counts, and one-hot encoded operational categories.
* **Notebook Link:** View the full implementation on [Kaggle](https://www.kaggle.com/code/cliaa74/hotel-booking-demand)

🛠️ Data Pipeline & Key Technical Learnings

1. Clean Feature Separation (Anti-Data-Leakage)
To ensure the integrity of the machine learning model, the target variable `is_canceled` is strictly dropped from the feature matrix `X` before performing the `train_test_split`. This prevents the model from directly reading the solution during training, ensuring a production-ready evaluation structure.

2. Feature Engineering & Selection
The features were meticulously prepared and engineered from the raw booking data to capture guest behavior and financial metrics:
* **Volume & Group Metrics:** Combined variables to create `total_nights` (weekend + week nights) and `total_pax` (adults + children + babies) to capture the true size of the booking.
* **Financial & Behavioral:** Engineered `revenue` and mapped structural indicators like `room_type_changes`, `has_agent`, and `has_company`.
* **Categorical Encodings:** Applied wide-scale encoding across key categorical indicators such as `deposit_type` (e.g., Non Refund), `market_segment` (e.g., Online TA), and geographic groups.

3. Structural Alignment & Feature Importance
Because tree-based ensemble models process tabular data strictly as mathematical matrices, analyzing feature importances exposes the primary drivers of the target variable. The pipeline validates that variables like `adr` (Average Daily Rate) and `deposit_type_Non Refund` carry the highest statistical weight when partitioning nodes within the Random Forest decision paths.

📂 Repository File Structure

* **hotel_booking_demand.ipynb:** The complete end-to-end notebook featuring data processing, feature engineering, categorical dummy encoding, matrix splitting, and Random Forest model evaluation.

🚀 Environment & Libraries

* **Python 3**
* **Pandas & NumPy** (Data manipulation, matrix transformations, and feature isolation)
* **Matplotlib & Seaborn** (Exploratory visualization and distribution tracking)
* **Scikit-Learn** (Train-test splits, Random Forest classifier, and classification report metrics)
