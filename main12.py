import tkinter as tk
from tkinter import messagebox

# Placeholder for your machine learning model's prediction function
def predict_cancer():
    try:
        # Get values from the entry fields   
        feature_1 = float(entry_feature_1.get())
        feature_2 = float(entry_feature_2.get())
        feature_3 = float(entry_feature_3.get())
        feature_4 = float(entry_feature_4.get())
        feature_5 = float(entry_feature_5.get())

        # Example input for the ML model (replace with actual prediction logic)
        user_data = [feature_1, feature_2, feature_3, feature_4, feature_5]
        
        # Dummy prediction (you can replace this with your trained model's output)
        prediction = "Cancerous" if sum(user_data) > 15 else "Non-Cancerous"
        confidence = 0.87  # Example confidence score
        
        # Display the result
        messagebox.showinfo("Prediction Result", f"Prediction: {prediction}\nConfidence: {confidence * 100:.2f}%")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all features.")

def reset_form():
    # Clear all entry fields
    entry_feature_1.delete(0, tk.END)
    entry_feature_2.delete(0, tk.END)
    entry_feature_3.delete(0, tk.END)
    entry_feature_4.delete(0, tk.END)
    entry_feature_5.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("TumoTrack - Cancer Detection System")
root.state('zoomed')  # Maximized window

# Title section
title_frame = tk.Frame(root, bg="#023E8A", height=80)
title_frame.pack(fill=tk.X)

title_label = tk.Label(
    title_frame,
    text="TumoTrack - Cancer Detection System",
    font=("Noto Serif", 24, "bold"),
    bg="#023E8A",
    fg="white"
)
title_label.pack(pady=22)


# Define fonts
heading_font = ("Noto Serif", 24)
text_font = ("Poppins", 16)
bg_color = "#FFE3E3"
# root.configure(bg=bg_color)

# Load and display the cancer icon (ensure you have an image in the same directory)
# Replace 'cancer_icon.png' with the path to your cancer icon
"""try:
    icon_image = Image.open("cancer_icon.png")  # Use an actual image file path
    icon_image = icon_image.resize((40, 40), Image.ANTIALIAS)  # Resize the image to fit
    icon_photo = ImageTk.PhotoImage(icon_image)
except Exception as e:
    print(f"Error loading image: {e}")
    icon_photo = None"""

# Create the labels and entry fields for the features
#tk.Label(root, text="TumoTrack", font=heading_font).pack(pady=10)
tk.Label(root, text="Enter Tumor Features", font=("Noto Serif", 20,"bold")).pack(pady=25)


tk.Label(root, text="Radius:", font=text_font).pack(pady=5)
entry_feature_1 = tk.Entry(root, font=text_font)
entry_feature_1.pack(pady=5)

tk.Label(root, text="Texture:", font=text_font).pack(pady=5)
entry_feature_2 = tk.Entry(root, font=text_font)
entry_feature_2.pack(pady=5)

tk.Label(root, text="Perimeter:", font=text_font).pack(pady=5)
entry_feature_3 = tk.Entry(root, font=text_font)
entry_feature_3.pack(pady=5)

tk.Label(root, text="Area:", font=text_font).pack(pady=5)
entry_feature_4 = tk.Entry(root, font=text_font)
entry_feature_4.pack(pady=5)

tk.Label(root, text="Smoothness:", font=text_font).pack(pady=5)
entry_feature_5 = tk.Entry(root, font=text_font)
entry_feature_5.pack(pady=5)


# Create a Frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=50)

# Create the "Predict" button (larger size)
predict_button = tk.Button( button_frame, text="Predict", command=predict_cancer, bg="#0077B6", fg="white", font=("Poppins", 16), width=15, height=2,borderwidth=1)
#predict_button.pack(pady=8)
predict_button.pack(side=tk.LEFT, padx=40)

# Create the "Check New" button to reset the form
check_new_button = tk.Button( button_frame, text="Check New", command=reset_form, bg="#46923c", fg="white", font=("Poppins", 16), width=15, height=2,borderwidth=1)
#check_new_button.pack(pady=8)
check_new_button.pack(side=tk.LEFT, padx=40) 

# Create the "Quit" button (larger size)
quit_button = tk.Button( button_frame, text="Quit", command=root.quit, bg="#d1001f", fg="white", font=("Poppins", 16), width=15, height=2,borderwidth=1)
#quit_button.pack(pady=8)
quit_button.pack(side=tk.LEFT, padx=40)


# Function to handle the Enter key press
def focus_next_entry(event, next_entry):
    next_entry.focus_set()
# Bind the Enter key to each Entry widget
entry_feature_1.bind("<Return>", lambda event: focus_next_entry(event,entry_feature_2))
entry_feature_2.bind("<Return>", lambda event: focus_next_entry(event, entry_feature_3))
entry_feature_3.bind("<Return>", lambda event: focus_next_entry(event, entry_feature_4))
entry_feature_4.bind("<Return>", lambda event: focus_next_entry(event, entry_feature_5))
entry_feature_5.bind("<Return>", lambda event: focus_next_entry(event, entry_feature_1))

# Start the GUI event loop
root.mainloop()
