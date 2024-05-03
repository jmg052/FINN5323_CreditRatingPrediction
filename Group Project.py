#%%
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import numpy as np
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs (1: INFO, 2: WARNING, 3: ERROR)

# Sample dataset loading - replace with your own data loading mechanism
data = pd.read_csv(r'path')
#%%
# Preprocessing data
# Drop identifiers and non-predictive columns
features = data.drop(['Rating Agency', 'Corporation', 'Rating', 'Rating Date', 'Ticker', 'Sector', 'CIK', 'SIC Code'], axis=1)
labels = data['Rating']

# Instantiate the encoder
label_encoder = LabelEncoder()

# Encode the labels to integers
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training and test sets using encoded labels
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Train the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)

# This dictionary maps strings to the actual model objects
models = {'Random Forest': model, 'XGBoost': xgb_model}

# Predict on the test set once and use the predictions for both metrics
y_pred = model.predict(X_test_scaled)

# After fitting the model, calculate its accuracy on the test set using the stored predictions
test_accuracy = accuracy_score(y_test, y_pred)
# Within your show_model_statistics function
classification_rep = classification_report(y_test, y_pred, zero_division=0,)

#%%
# Update Neural Network model creation (remove 'input_shape' from first Dense layer):
def create_neural_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Use Input layer to define input shape
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate the neural network model
nn_model = create_neural_network(X_train_scaled.shape[1], len(label_encoder.classes_))

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity.
)

# Train the model with the early stopping callback
nn_model.fit(
    X_train_scaled, 
    y_train, 
    epochs=100,  # Start with 100 epochs, early stopping may halt training before reaching 100
    batch_size=32,  # A common starting point for batch size
    verbose=1, 
    validation_split=0.2,  # Use part of the training data for validation
    callbacks=[early_stopping]  # Include early stopping in the callbacks
)

#%%
def preprocess_input(selected_stock):
    # Fetch the stock's features, excluding the 'Sector', 'CIK', 'SIC Code', and other non-predictive columns
    stock_features = data[data['Ticker'] == selected_stock].drop(['Rating Agency', 'Corporation', 'Rating', 'Rating Date', 'Ticker', 'Sector', 'CIK', 'SIC Code'], axis=1)
    # Ensure no additional columns that weren't in the training data
    if set(stock_features.columns) != set(X_train.columns):
        raise ValueError("Features of input data do not match features of training data.")
    # Scale the features similarly to the training data
    stock_features_scaled = scaler.transform(stock_features)
    return stock_features_scaled


# Function to predict the credit rating with the selected model
def predict_credit_rating(selected_stock, selected_model):
    stock_features_scaled = preprocess_input(selected_stock)
    if selected_model == 'Neural Network':
        nn_y_pred_probs = nn_model.predict(stock_features_scaled)
        nn_y_pred = nn_y_pred_probs.argmax(axis=1)
        string_prediction = label_encoder.inverse_transform(nn_y_pred)
    else:
        numeric_prediction = models[selected_model].predict(stock_features_scaled)
        string_prediction = label_encoder.inverse_transform(numeric_prediction)
    prediction_label.config(text=f"Predicted Credit Rating with {selected_model}: {string_prediction[0]}")

def show_model_statistics(selected_model):
    if selected_model in ['Random Forest', 'XGBoost']:
        y_pred = models[selected_model].predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        # Ensure that classification report includes all classes and labels match the names from the encoder
        classification_rep = classification_report(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_, zero_division=0)
        update_report_text(selected_model, test_accuracy, classification_rep)
    elif selected_model == 'Neural Network':
        nn_y_pred_probs = nn_model.predict(X_test_scaled)
        nn_y_pred = nn_y_pred_probs.argmax(axis=1)
        test_accuracy_nn = accuracy_score(y_test, nn_y_pred)
        # Ensure to include all potential labels
        classification_rep_nn = classification_report(y_test, nn_y_pred, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_, zero_division=0)
        update_report_text('Neural Network', test_accuracy_nn, classification_rep_nn)

#%%
# Tkinter GUI code
root = tk.Tk()
root.title("Stock Credit Rating Predictor")
root.geometry('512x768')  

# Create a notebook widget for tabs
notebook = ttk.Notebook(root)

# Create a frame for each model's statistics
rf_frame = ttk.Frame(notebook)  # Frame for Random Forest statistics
xgb_frame = ttk.Frame(notebook)  # Frame for XGBoost statistics
notebook.add(rf_frame, text='Random Forest')
notebook.add(xgb_frame, text='XGBoost')

# Add a frame for Neural Network statistics
nn_frame = ttk.Frame(notebook)  # Frame for Neural Network statistics
notebook.add(nn_frame, text='Neural Network')

# Function to be called when the tab is changed
def on_tab_selected(event):
    selected_tab = event.widget.tab(event.widget.index("current"))['text']
    show_model_statistics(selected_tab)

# Bind the on_tab_selected function to the notebook to handle tab changes
notebook.bind("<<NotebookTabChanged>>", on_tab_selected)

notebook.pack(expand=True, fill='both')

# Move the existing stock selection to the main window
stock_label = ttk.Label(root, text="Select a stock:")
stock_label.pack(pady=5)

stocks_combobox = ttk.Combobox(root, values=data['Ticker'].unique().tolist())
stocks_combobox.pack()

# Update the predict button command to use the tab label as the selected model
predict_button = ttk.Button(root, text="Predict Credit Rating", command=lambda: predict_credit_rating(stocks_combobox.get(), notebook.tab(notebook.index("current"), "text")))
predict_button.pack(pady=10)

prediction_label = ttk.Label(root, text="Predicted Credit Rating: None")
prediction_label.pack(pady=10)

# Now create Text widgets with scrollbars for the classification report inside each frame
rf_report_text = tk.Text(rf_frame, height=15, width=45)  # Reduced width from 90 to 45
rf_report_scroll = ttk.Scrollbar(rf_frame, command=rf_report_text.yview)
rf_report_text.configure(yscrollcommand=rf_report_scroll.set)
rf_report_scroll.pack(side='right', fill='y')
rf_report_text.pack(side='left', fill='both', expand=True)

xgb_report_text = tk.Text(xgb_frame, height=15, width=45)  # Reduced width from 90 to 45
xgb_report_scroll = ttk.Scrollbar(xgb_frame, command=xgb_report_text.yview)
xgb_report_text.configure(yscrollcommand=xgb_report_scroll.set)
xgb_report_scroll.pack(side='right', fill='y')
xgb_report_text.pack(side='left', fill='both', expand=True)

# Create Text widget with scrollbar for the Neural Network classification report
nn_report_text = tk.Text(nn_frame, height=15, width=45)  # Reduced width from 90 to 45
nn_report_scroll = ttk.Scrollbar(nn_frame, command=nn_report_text.yview)
nn_report_text.configure(yscrollcommand=nn_report_scroll.set)
nn_report_scroll.pack(side='right', fill='y')
nn_report_text.pack(side='left', fill='both', expand=True)

# Create a status bar at the bottom of the root window
status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side='bottom', fill='x')

def update_report_text(model_name, accuracy, report):
    if model_name == 'Random Forest':
        rf_report_text.delete('1.0', tk.END)
        rf_report_text.insert(tk.END, report)
        text_widget = rf_report_text
        status_bar.config(text=f"Random Forest Model Accuracy: {accuracy:.2f}")
    elif model_name == 'XGBoost':
        xgb_report_text.delete('1.0', tk.END)
        text_widget = xgb_report_text
        xgb_report_text.insert(tk.END, report)
        status_bar.config(text=f"XGBoost Model Accuracy: {accuracy:.2f}")
    elif model_name == 'Neural Network':
        nn_report_text.delete('1.0', tk.END)
        nn_report_text.insert(tk.END, report)
        text_widget = nn_report_text
        status_bar.config(text=f"Neural Network Model Accuracy: {accuracy:.2f}")
    else:
            return  # Invalid model name
    
    # Update the text widget and status bar
    text_widget.delete('1.0', tk.END)
    text_widget.insert(tk.END, report)
    status_bar.config(text=f"{model_name} Model Accuracy: {accuracy:.2f}")

# Initially show Random Forest statistics
notebook.select(rf_frame)
show_model_statistics('Random Forest')

root.mainloop()
