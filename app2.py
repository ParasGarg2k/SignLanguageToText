# Importing Libraries
import numpy as np
import cv2
import os
import operator
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
import enchant
import json
import tensorflow as tf
from tensorflow import keras
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Application
from autocorrect import Speller
from spellchecker import SpellChecker
class Application:
    def __init__(self):
        self.hs = Speller(lang='en')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load main model
        self.loaded_model = keras.models.load_model("Models/model_new1.keras")
        self.loaded_model_dru = keras.models.load_model("Models/model_new1 copy.keras")
        self.loaded_model_tkdi = keras.models.load_model("Models/model_new1 copy 2.keras")
        self.loaded_model_smn = keras.models.load_model("Models/model_new1 copy 3.keras")
        print("Loaded all models from disk")

        # Initialize counters
        self.ct = {letter: 0 for letter in ascii_uppercase}
        self.ct["blank"] = 0
        self.blank_flag = 0

        # Setup GUI
        self.setup_gui()
        
        # Initialize variables
        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        
        # Start video loop
        self.video_loop()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("900x900")

        # Main video panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        # Processed image panel
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        # Title
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        # Character display
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        # Word display
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        # Sentence display
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        # Suggestions
        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        # Suggestion buttons
        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)
        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)
        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)

            # Define ROI
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            # Draw rectangle for ROI
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            
            # Convert image for display
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Process ROI
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Predict and update
            self.predict(res)

            # Update processed image display
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            # Update text displays
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Update suggestions
            
            spell = SpellChecker()
            predicts = list(spell.candidates(self.word)) 
            
            # Update suggestion buttons
            self.update_suggestion_buttons(predicts)

        self.root.after(5, self.video_loop)

    def update_suggestion_buttons(self, predicts):
        buttons = [self.bt1, self.bt2, self.bt3]
        for i, button in enumerate(buttons):
            if len(predicts) > i:
                button.config(text=predicts[i], font=("Courier", 20))
            else:
                button.config(text="")
  



    def predict(self, test_image):
        # Ensure test_image is grayscale
        try:
        # Ensure test_image is grayscale
            if len(test_image.shape) == 3:  # If the image has 3 channels (RGB)
                test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # Resize to the expected input size (128x128)
            test_image = cv2.resize(test_image, (128, 128))

            # Normalize pixel values (important for CNNs)
            test_image = test_image / 255.0

            # Reshape to match model's expected input shape
            test_image = test_image.reshape(1, 128, 128, 1)  # (batch_size, height, width, channels)

            # Debugging: Check shape before prediction
            print("Test Image Shape:", test_image.shape)

            # Get predictions from all models
            result = self.loaded_model.predict(test_image)  
            result_dru = self.loaded_model_dru.predict(test_image)
            result_tkdi = self.loaded_model_tkdi.predict(test_image)
            result_smn = self.loaded_model_smn.predict(test_image)

            # Debugging: Print output shapes
            print(f"Model Output Shape: {result.shape}")

            # Ensure model output is correctly reshaped
            if len(result.shape) > 2:
                result = result.reshape(1, -1)  # Ensure it's (1, num_classes)

            # Validate output shape
            if result.shape != (1, 27):  # Expected output shape: (batch_size, 27)
                print(f"⚠️ Warning: Unexpected output shape {result.shape}, setting to 'blank'")
                self.current_symbol = "blank"
                return

            # Get the most probable class index
            prediction_index = int(np.argmax(result[0]))

            # Ensure valid index range
            if prediction_index < 0 or prediction_index >= 27:
                print(f"⚠️ Warning: Invalid index {prediction_index}, setting to 'blank'")
                self.current_symbol = "blank"
                return  # Exit early to avoid further errors

            # Convert index to character
            self.current_symbol = "blank" if prediction_index == 0 else ascii_uppercase[prediction_index - 1]

            # Layer 2 - DRU classification
            if self.current_symbol in ["D", "R", "U"]:
                dru_index = np.argmax(result_dru[0])
                self.current_symbol = ["D", "R", "U"][dru_index]

            # Layer 2 - TKDI classification
            if self.current_symbol in ["D", "I", "K", "T"]:
                tkdi_index = np.argmax(result_tkdi[0])
                self.current_symbol = ["D", "I", "K", "T"][tkdi_index]

            # Layer 2 - SMN classification
            if self.current_symbol in ["M", "N", "S"]:
                smn_index = np.argmax(result_smn[0])
                self.current_symbol = ["M", "N", "S"][smn_index]

            # Update counter and handle blank detection
            self.update_counter()

        except Exception as e:
            print(f"⚠️ Error in prediction: {str(e)}")
            self.current_symbol = "blank"



    def update_counter(self):
        if self.current_symbol == "blank":
            self.ct = {letter: 0 for letter in ascii_uppercase}
        else:
            self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            # Check for potential conflicts
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                if abs(self.ct[self.current_symbol] - self.ct[i]) <= 20:
                    self.reset_counters()
                    return

            # Handle word formation
            self.process_symbol()

    def process_symbol(self):
        self.reset_counters()
        
        if self.current_symbol == "blank":
            if self.blank_flag == 0:
                self.blank_flag = 1
                if len(self.str) > 0:
                    self.str += " "
                self.str += self.word
                self.word = ""
        else:
            if len(self.str) > 16:
                self.str = ""
            self.blank_flag = 0
            self.word += self.current_symbol

    def reset_counters(self):
        self.ct = {letter: 0 for letter in ascii_uppercase}
        self.ct["blank"] = 0

    def action1(self):
        self.process_suggestion(0)

    def action2(self):
        self.process_suggestion(1)

    def action3(self):
        self.process_suggestion(2)

    def process_suggestion(self, index):
        spell = SpellChecker()
        predicts = list(spell.candidates(self.word)) 
        if len(predicts) > index:
            self.word = ""
            self.str += " "
            self.str += predicts[index]

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Application...")
    try:
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {str(e)}")