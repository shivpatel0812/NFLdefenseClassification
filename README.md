
# NFL Defensive Formations Image Classification

This project demonstrates the process of downloading NFL defensive formation images and classifying them using a Support Vector Machine (SVM) model. The project was developed using Python and executed in Google Colab. The model is trained on various defensive formations in the NFL and predicts the formation based on input images.

## Features

- **Image Downloading**: Using the Bing Image Downloader, the project downloads images of different NFL defensive formations such as:
  - Quarter defense formation
  - Man-to-man defense formation
  - Zone blitz defense formation
  - Dime defense formation
  - Blitz defense formation
  - Base defense formation
  - Nickel defensive formation
- **Image Preprocessing**: The images are resized, flattened, and prepared for training.
- **Image Classification**: An SVM model is trained to classify the images into one of the defensive formations using Scikit-learn's SVM and GridSearchCV.
- **Model Evaluation**: The accuracy of the model is tested using a confusion matrix and accuracy score.
- **Real-Time Image Prediction**: Users can input an image URL, and the model will predict the corresponding NFL defensive formation.
- **Deployment**: The model is ready to be deployed with Streamlit and Ngrok.

## Technologies

- **Languages**: Python
- **Libraries**: 
  - `bing-image-downloader` for downloading images
  - `scikit-image` for image preprocessing
  - `matplotlib` for visualizations
  - `sklearn` for training and evaluating the SVM model
  - `streamlit` for web-based user interface
  - `pyngrok` for making the web application public

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd nfl_defense_image_classification
   ```

2. Install the necessary Python packages:
   ```bash
   pip install ipython-autotime bing-image-downloader scikit-image matplotlib scikit-learn streamlit pyngrok
   ```

3. Download NFL defensive formation images using Bing Image Downloader:
   The code already includes pre-configured categories of NFL defensive formations. You can run the script to download the images:
   ```python
   from bing_image_downloader import downloader
   downloader.download("Quarter defnese formation nfl", limit=30, output_dir="images", adult_filter_off=True)
   ```

4. Preprocess and Train the Model:
   - Load the images, resize them to (150, 150), and flatten them to feed into the SVM.
   - Use `GridSearchCV` to optimize SVM hyperparameters (e.g., `C` and `gamma`).
   - Split the dataset into training and testing sets.

5. Evaluate the Model:
   - Use `accuracy_score` and `confusion_matrix` to evaluate the model performance on the test set.

6. Real-Time Prediction:
   - Input an image URL, and the model will predict the corresponding defensive formation.

7. Deploy with Streamlit and Ngrok:
   - Use Streamlit to create a web-based interface for users to upload images and view predictions.
   - Use Ngrok to make the Streamlit app public.


