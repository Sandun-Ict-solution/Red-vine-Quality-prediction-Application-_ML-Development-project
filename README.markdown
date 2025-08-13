                                                🍷 Wine Quality Predictor

A Streamlit-based web application to predict red wine quality using machine learning. The app leverages a trained model to classify wines as **Good** (quality ≥ 6) or **Bad** (quality &lt; 6) based on chemical properties, offering interactive data exploration, visualizations, and model performance insights.

## 📋 Features

- **Home Page**: Overview of the dataset with key metrics and a sample of the wine data.
- <img width="1918" height="1066" alt="1" src="https://github.com/user-attachments/assets/4818bade-c0aa-42b7-995a-161936d688ba" />

- **Data Explorer**: Filter and explore the dataset with interactive sliders for alcohol content, pH, and quality ratings.
- <img width="1912" height="1078" alt="2" src="https://github.com/user-attachments/assets/543744c5-d5fb-4616-bd2d-0526f5c64b46" />

- **Visualizations**: Interactive plots including box plots, correlation heatmaps, histograms, violin plots, and scatter plots to analyze wine features.
- <img width="1906" height="1078" alt="3" src="https://github.com/user-attachments/assets/51582109-636f-45d0-a97f-0b007028dc01" />

- **Predict Quality**: Input wine characteristics to predict quality with confidence scores and a probability gauge.
- <img width="1912" height="1078" alt="4" src="https://github.com/user-attachments/assets/69f90092-d767-426b-8276-e3fbd2b7bbd7" />

- **Model Performance**: Compare model metrics (train/test accuracy) and view feature importance rankings.
<img width="1918" height="1078" alt="5" src="https://github.com/user-attachments/assets/2dc46193-3ef8-4fdb-a9fb-8fa227bf5cc7" />

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web app framework for interactive UI
- **Pandas & NumPy**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Joblib**: Model loading and caching
- **Seaborn & Matplotlib**: Additional plotting utilities (used in model training)

## 📊 Dataset

The app uses the **Red Wine Quality** dataset (`winequality-red.csv`), which contains 1,599 wine samples with 11 chemical features and a quality rating (3–8). A binary quality label is derived (Good: ≥6, Bad: &lt;6).

### Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/wine-quality-predictor.git
   ```
2. Navigate to the project directory:

   ```bash
   cd wine-quality-predictor
   ```
3. Install required packages:

   ```bash
   pip install streamlit pandas numpy plotly joblib seaborn matplotlib
   ```
4. Ensure the dataset (`winequality-red.csv`) and trained model (`model.pkl`) are in the project directory. Run `model_training.py` to generate the model and related files (`model_comparison.csv`, `feature_importance.csv`).

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📈 Usage

1. **Navigate**: Use the sidebar to switch between Home, Data Explorer, Visualizations, Predict Quality, and Model Performance.
2. **Explore Data**: Filter wines by alcohol, pH, and quality, and view statistical summaries.
3. **Visualize**: Analyze feature distributions, correlations, and relationships with interactive plots.
4. **Predict**: Enter wine characteristics to get a quality prediction with confidence scores.
5. **Evaluate Model**: Review model accuracy and feature importance for insights into the prediction process.

## 📂 File Structure

- `app.py`: Main Streamlit application
- `winequality-red.csv`: Dataset file
- `model.pkl`: Trained machine learning model
- `model_comparison.csv`: Model performance metrics
- `feature_importance.csv`: Feature importance rankings
- `model_training.py`: Script to train the model (not included in this repo)

## 🛑 Notes

- Ensure all required files (`winequality-red.csv`, `model.pkl`, etc.) are present in the directory before running.
- Run `model_training.py` to generate the model and supporting files if they are missing.
- The app uses caching (`@st.cache_data`, `@st.cache_resource`) to optimize performance.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request

## 📜 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or feedback, reach out via ssandu809@gmail.comor open an issue on GitHub.

---

**Cheers to great wine and great code! 🍷**

                                           **Developed by Sandun Wijesingha**
