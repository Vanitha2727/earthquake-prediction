import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, ttk
import folium
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EarthquakePredictionUI:
    def __init__(self, master):
        self.master = master
        self.master.title("QuakeSense: Advanced Earthquake Prediction System")
        self.master.geometry("800x600")
        self.master.configure(bg="#2E4057")  # Dark blue background
        
        # Create a main frame
        main_frame = tk.Frame(self.master, bg="#2E4057", padx=20, pady=20)
        main_frame.pack(expand=True, fill="both")
        
        # Project title with better styling
        self.title_frame = tk.Frame(main_frame, bg="#2E4057")
        self.title_frame.pack(pady=20)
        
        self.title_label = tk.Label(
            self.title_frame, 
            text="QuakeSense", 
            font=("Helvetica", 28, "bold"),
            fg="#F8F4E3",  # Light cream text
            bg="#2E4057"   # Dark blue background
        )
        self.title_label.pack()
        
        self.subtitle_label = tk.Label(
            self.title_frame, 
            text="Advanced Earthquake Prediction System",
            font=("Helvetica", 16),
            fg="#F8F4E3",  # Light cream text
            bg="#2E4057"   # Dark blue background
        )
        self.subtitle_label.pack(pady=5)
        
        # Logo or icon placeholder
        self.logo_frame = tk.Frame(main_frame, bg="#2E4057", height=150, width=150)
        self.logo_frame.pack(pady=30)
        
        self.logo_canvas = tk.Canvas(self.logo_frame, height=150, width=150, bg="#2E4057", highlightthickness=0)
        self.logo_canvas.pack()
        
        # Draw a simple earthquake wave icon
        self.logo_canvas.create_line(25, 75, 45, 95, 65, 55, 85, 95, 105, 75, 125, 95, fill="#FF6B6B", width=3)
        self.logo_canvas.create_oval(70, 70, 80, 80, fill="#FF6B6B", outline="#FF6B6B")
        
        # Continue button with better styling
        self.continue_button = tk.Button(
            main_frame, 
            text="Start Prediction",
            font=("Helvetica", 12, "bold"),
            command=self.show_prediction_page,
            bg="#FF6B6B",  # Coral red button
            fg="white",
            activebackground="#E05555",
            activeforeground="white",
            relief=tk.RAISED,
            bd=2,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.continue_button.pack(pady=20)
        
        # Add version info
        version_label = tk.Label(
            main_frame, 
            text="v1.0.0", 
            font=("Helvetica", 8),
            fg="#F8F4E3",
            bg="#2E4057"
        )
        version_label.pack(side=tk.BOTTOM, pady=10)

    def show_prediction_page(self):
        self.master.destroy()
        prediction_window = tk.Tk()
        PredictionPage(prediction_window)
        prediction_window.mainloop()

class PredictionPage:
    def __init__(self, master):
        self.master = master
        self.master.title("QuakeSense: Earthquake Prediction")
        self.master.geometry("1000x800")
        self.master.configure(bg="#2E4057")  # Dark blue background
        
        # Create a header frame
        header_frame = tk.Frame(self.master, bg="#1D2D44", padx=10, pady=10)
        header_frame.pack(fill="x")
        
        # Add logo and title to header
        header_title = tk.Label(
            header_frame, 
            text="QuakeSense",
            font=("Helvetica", 16, "bold"),
            fg="#F8F4E3",
            bg="#1D2D44"
        )
        header_title.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = tk.Frame(self.master, bg="#2E4057", padx=20, pady=20)
        content_frame.pack(fill="both", expand=True)
        
        # Map section
        self.map_label = tk.Label(
            content_frame, 
            text="Global Earthquake Map",
            font=("Helvetica", 14, "bold"),
            fg="#F8F4E3",
            bg="#2E4057"
        )
        self.map_label.pack(pady=(0, 10))
        
        self.map_frame = tk.Frame(content_frame, bg="#F8F4E3", height=400)
        self.map_frame.pack(side="top", fill="both", expand=True, pady=10)
        
        self.map = folium.Map(location=[0, 0], zoom_start=2)
        self.map.save("map.html")
        self.map_view = tk.Label(self.map_frame, text="Map will be displayed here after data is loaded", 
                                 bg="#F8F4E3", fg="#333333")
        self.map_view.pack(expand=True)
        
        # Button frame
        button_frame = tk.Frame(content_frame, bg="#2E4057", pady=15)
        button_frame.pack(fill="x")
        
        # Style for buttons (using ttk for better looking buttons)
        style = ttk.Style()
        style.configure("TButton", 
                        font=("Helvetica", 12), 
                        background="#0000FF", 
                        foreground="red")
        
        self.upload_button = ttk.Button(
            button_frame, 
            text="Upload Dataset",
            style="TButton",
            command=self.load_dataset
        )
        self.upload_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = ttk.Button(
            button_frame, 
            text="Run Prediction",
            style="TButton",
            command=self.predict
        )
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_frame = tk.Frame(self.master, bg="#1D2D44", height=25)
        self.status_frame.pack(side=tk.BOTTOM, fill="x")
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="Ready to load data",
            font=("Helvetica", 10),
            fg="#F8F4E3",
            bg="#1D2D44",
            anchor="w",
            padx=10
        )
        self.status_label.pack(side=tk.LEFT)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Earthquake Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.status_label.config(text=f"Dataset loaded: {len(self.data)} records found")
                messagebox.showinfo("Success", f"Dataset loaded successfully with {len(self.data)} records!")
            except Exception as e:
                self.status_label.config(text="Error loading dataset")
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")

    def preprocess_data(self):
        if hasattr(self, 'data'):
            try:
                self.status_label.config(text="Preprocessing data...")
                self.master.update()
                
                self.data['Datetime'] = pd.to_datetime(self.data['Date'] + ' ' + self.data['Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                self.data.dropna(subset=['Datetime'], inplace=True)
                self.data['Year'] = self.data['Datetime'].dt.year
                self.data['Month'] = self.data['Datetime'].dt.month
                self.data['Day'] = self.data['Datetime'].dt.day
                self.data['Hour'] = self.data['Datetime'].dt.hour
                self.data.drop(columns=['Date', 'Time', 'Datetime'], inplace=True)
                self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
                le = LabelEncoder()
                self.data['Type'] = le.fit_transform(self.data['Type'])
                self.data['Magnitude Type'] = le.fit_transform(self.data['Magnitude Type'])
                self.X = self.data.drop(columns=['Magnitude'])
                self.y = self.data['Magnitude']
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                self.scaler = StandardScaler()
                self.X_train_scaled = self.scaler.fit_transform(self.X_train)
                self.X_test_scaled = self.scaler.transform(self.X_test)
                
                self.status_label.config(text="Data preprocessing complete")
                messagebox.showinfo("Success", "Data preprocessed successfully!")
                return True
            except Exception as e:
                self.status_label.config(text="Error in preprocessing")
                messagebox.showerror("Error", f"Error preprocessing data: {str(e)}")
                return False
        else:
            self.status_label.config(text="No dataset loaded")
            messagebox.showerror("Error", "No dataset loaded! Please upload a dataset first.")
            return False

    def predict(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "No dataset loaded! Please upload a dataset first.")
            return
            
        if self.preprocess_data():
            try:
                self.status_label.config(text="Running prediction model...")
                self.master.update()
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(self.X_train_scaled, self.y_train)
                y_pred_rf = rf.predict(self.X_test_scaled)
                
                self.status_label.config(text="Prediction complete")

                # Create a new window for displaying predictions and plots
                prediction_window = Toplevel(self.master)
                prediction_window.title("QuakeSense: Prediction Results")
                prediction_window.geometry("800x600")
                prediction_window.configure(bg="#F8F4E3")  # Light cream background

                # Create a header
                header = tk.Frame(prediction_window, bg="#1D2D44", padx=10, pady=10)
                header.pack(fill="x")
                
                header_title = tk.Label(
                    header, 
                    text="Earthquake Prediction Results",
                    font=("Helvetica", 16, "bold"),
                    fg="#F8F4E3",
                    bg="#1D2D44"
                )
                header_title.pack()
                
                # Content frame
                content = tk.Frame(prediction_window, bg="#F8F4E3", padx=20, pady=20)
                content.pack(fill="both", expand=True)

                # Display prediction summary
                summary_frame = tk.Frame(content, bg="#F8F4E3")
                summary_frame.pack(fill="x", pady=10)
                
                predictions_label = tk.Label(
                    summary_frame, 
                    text="Prediction Summary", 
                    font=("Helvetica", 14, "bold"),
                    bg="#F8F4E3",
                    fg="#333333"
                )
                predictions_label.pack(pady=5)
                
                avg_pred = sum(y_pred_rf) / len(y_pred_rf)
                max_pred = max(y_pred_rf)
                
                summary_text = f"Average Predicted Magnitude: {avg_pred:.2f}\n"
                summary_text += f"Maximum Predicted Magnitude: {max_pred:.2f}\n"
                summary_text += f"Total Predictions: {len(y_pred_rf)}"
                
                summary_info = tk.Label(
                    summary_frame,
                    text=summary_text,
                    font=("Helvetica", 12),
                    bg="#F8F4E3",
                    fg="#333333",
                    justify=tk.LEFT
                )
                summary_info.pack(pady=5)

                # Display plots
                self.display_plots(content, y_pred_rf)
                
                # Button to return to main window
                return_button = ttk.Button(
                    content, 
                    text="Back to Main Window", 
                    command=prediction_window.destroy
                )
                return_button.pack(pady=10)

            except Exception as e:
                self.status_label.config(text="Error in prediction")
                messagebox.showerror("Error", f"Error predicting: {str(e)}")

    def display_plots(self, parent_frame, y_pred_rf):
        try:
            # Create a frame for the plot
            plot_frame = tk.Frame(parent_frame, bg="#F8F4E3")
            plot_frame.pack(fill="both", expand=True, pady=10)
            
            # Plot actual vs predicted
            fig_actual_pred = plt.figure(figsize=(8, 5))
            plt.plot(self.y_test.values[:50], label='Actual Magnitude', color='#1D2D44', linewidth=2)
            plt.plot(y_pred_rf[:50], label='Predicted Magnitude', color='#FF6B6B', linewidth=2)
            plt.xlabel('Sample Index')
            plt.ylabel('Magnitude')
            plt.title('Actual vs Predicted Earthquake Magnitude')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            fig_actual_pred.patch.set_facecolor('#F8F4E3')
            plt.tight_layout()

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig_actual_pred, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error displaying plots: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EarthquakePredictionUI(root)
    root.mainloop()