import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle

class BreastCancerPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("TUMO TRACK")
        self.root.geometry("1200x800")
        self.root.configure(bg='#FFC0CB')

        # Load the model and scaler
        try:
            self.model = pickle.load(open('breast_cancer.pkl', 'rb'))
            self.scaler = pickle.load(open('scaler.pkl', 'rb'))
        except:
            messagebox.showerror("Error", "Model files not found!")

        # Sample values
        self.benign_sample = {
            'mean': [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766],
            'se': [0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023],
            'worst': [15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
        }
        
        self.malignant_sample = {
            'mean': [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871],
            'se': [1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193],
            'worst': [25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
        }

        self.create_widgets()

    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="TUMO TRACK", 
                        font=("Helvetica", 36, "bold"), bg='#FFC0CB', pady=20)
        title.pack()

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)

        # Center the main frame
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Style configuration for column headers
        style = ttk.Style()
        style.configure('Header.TLabelframe.Label', 
                        font=('Helvetica', 16, 'bold'))

        # Create three columns with larger font headers
        self.mean_frame = ttk.LabelFrame(main_frame, text="MEAN VALUES", style='Header.TLabelframe')
        self.se_frame = ttk.LabelFrame(main_frame, text="STANDARD ERROR VALUES", style='Header.TLabelframe')
        self.worst_frame = ttk.LabelFrame(main_frame, text="WORST VALUES", style='Header.TLabelframe')

        # Center align all frames
        self.mean_frame.grid(row=0, column=0, padx=10, pady=5, sticky='nsew')
        self.se_frame.grid(row=0, column=1, padx=10, pady=5, sticky='nsew')
        self.worst_frame.grid(row=0, column=2, padx=10, pady=5, sticky='nsew')

        # Configure equal column weights for centering
        main_frame.grid_columnconfigure(0, weight=1, uniform='column')
        main_frame.grid_columnconfigure(1, weight=1, uniform='column')
        main_frame.grid_columnconfigure(2, weight=1, uniform='column')

        # Create entry fields
        self.entries = {}
        self.create_entry_fields()

        # Create buttons frame
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(pady=(0, 10))

        # Sample data buttons
        ttk.Button(buttons_frame, text="Load Benign Sample", 
                  command=self.load_benign_sample).grid(row=0, column=0, padx=5)
        ttk.Button(buttons_frame, text="Load Malignant Sample", 
                  command=self.load_malignant_sample).grid(row=0, column=1, padx=5)
        ttk.Button(buttons_frame, text="Clear All", 
                  command=self.clear_all).grid(row=0, column=2, padx=5)

        # Predict button
        predict_btn = ttk.Button(self.root, text="PREDICT", command=self.predict)
        predict_btn.pack(pady=5)

        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results")
        results_frame.pack(padx=20, pady=5, fill='x')

        self.result_text = tk.StringVar()
        self.benign_prob = tk.StringVar()
        self.malignant_prob = tk.StringVar()

        ttk.Label(results_frame, textvariable=self.result_text, 
                 font=("Helvetica", 16)).pack(pady=5)
        ttk.Label(results_frame, textvariable=self.benign_prob, 
                 font=("Helvetica", 12)).pack(pady=2)
        ttk.Label(results_frame, textvariable=self.malignant_prob, 
                 font=("Helvetica", 12)).pack(pady=2)

    def create_entry_fields(self):
        features = {
            'mean': ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                    "Compactness", "Concavity", "Concave Points", "Symmetry", 
                    "Fractal Dimension"],
            'se': ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                  "Compactness", "Concavity", "Concave Points", "Symmetry", 
                  "Fractal Dimension"],
            'worst': ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                     "Compactness", "Concavity", "Concave Points", "Symmetry", 
                     "Fractal Dimension"]
        }

        frames = {
            'mean': self.mean_frame,
            'se': self.se_frame,
            'worst': self.worst_frame
        }

        for category, frame in frames.items():
            # Configure grid weights for centering within each frame
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_columnconfigure(1, weight=1)
            
            for i, feature in enumerate(features[category]):
                ttk.Label(frame, text=f"{feature}:").grid(row=i, column=0, padx=(20, 5), pady=2, sticky='e')
                entry = ttk.Entry(frame, width=15)
                entry.grid(row=i, column=1, padx=(5, 20), pady=2)
                self.entries[f"{category}_{feature.lower().replace(' ', '_')}"] = entry

    def load_benign_sample(self):
        self.load_sample(self.benign_sample)
        messagebox.showinfo("Sample Loaded", "Benign sample data has been loaded!")

    def load_malignant_sample(self):
        self.load_sample(self.malignant_sample)
        messagebox.showinfo("Sample Loaded", "Malignant sample data has been loaded!")

    def load_sample(self, sample_data):
        categories = ['mean', 'se', 'worst']
        features = ["radius", "texture", "perimeter", "area", "smoothness",
                   "compactness", "concavity", "concave_points", "symmetry", 
                   "fractal_dimension"]
        
        for cat_idx, category in enumerate(categories):
            for feat_idx, feature in enumerate(features):
                entry_key = f"{category}_{feature}"
                self.entries[entry_key].delete(0, tk.END)
                self.entries[entry_key].insert(0, str(sample_data[category][feat_idx]))

    def clear_all(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_text.set("")
        self.benign_prob.set("")
        self.malignant_prob.set("")

    def predict(self):
        try:
            # Collect input data
            input_data = []
            categories = ['mean', 'se', 'worst']
            features = ["radius", "texture", "perimeter", "area", "smoothness",
                       "compactness", "concavity", "concave_points", "symmetry", 
                       "fractal_dimension"]
            
            for category in categories:
                for feature in features:
                    value = float(self.entries[f"{category}_{feature}"].get())
                    input_data.append(value)

            # Scale features and predict
            features = np.array(input_data).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)

            # Display results
            self.result_text.set(f"Prediction: {'MALIGNANT' if prediction[0] == 1 else 'BENIGN'}")
            self.benign_prob.set(f"Benign Probability: {probabilities[0][0]:.2%}")
            self.malignant_prob.set(f"Malignant Probability: {probabilities[0][1]:.2%}")

        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerPredictor(root)
    root.mainloop()
