import tkinter as tk
from tkinter import filedialog, messagebox, Frame

class SimpleUI:
    def __init__(self, root: tk.Tk):
        self.root = root
    
        self.root.title('Dataset Reduction Test Suite')

        # Set default window size
        self.root.geometry('400x400')
        
        # Base csv upload button
        base_csv_frame=Frame(root, width=400, height=200)
        base_csv_frame.pack(fill='x', pady=10)
        self.base_upload_label = tk.Label(base_csv_frame, text='Base dataset (.csv):')
        self.base_upload_btn = tk.Button(base_csv_frame, text='Upload File', command=self.upload_base)
        self.base_upload_label.pack(side='left', padx=25)
        self.base_upload_btn.pack(side='right', padx=25)

        # Index csv upload button
        reduction_csv_frame=Frame(root, width=400, height=200)
        reduction_csv_frame.pack(fill='x', pady=10)
        self.reduction_upload_label = tk.Label(reduction_csv_frame, text='Reduction indices (.csv):')
        self.reduction_upload_btn = tk.Button(reduction_csv_frame, text='Upload File(s)', command=self.upload_reductions)
        self.reduction_upload_label.pack(side='left', padx=25)
        self.reduction_upload_btn.pack(side='right', padx=25)
        
        # Custom models upload button
        custom_models_frame=Frame(root, width=400, height=200)
        custom_models_frame.pack(fill='x', pady=10)
        self.model_upload_label = tk.Label(custom_models_frame, text='Custom Models (.keras) [Optional]:')
        self.model_upload_btn = tk.Button(custom_models_frame, text='Upload File(s)', command=self.upload_custom_models)
        self.model_upload_label.pack(side='left', padx=25)
        self.model_upload_btn.pack(side='right', padx=25)
        
        # Checkboxes
        self.check_var1 = tk.IntVar()
        self.check_var2 = tk.IntVar()
        self.checkbox1 = tk.Checkbutton(root, text='Option 1', variable=self.check_var1)
        self.checkbox2 = tk.Checkbutton(root, text='Option 2', variable=self.check_var2)
        self.checkbox1.pack(pady=5)
        self.checkbox2.pack(pady=5)
        
        # Text input field
        self.text_label = tk.Label(root, text='Text Input:')
        self.text_label.pack(pady=5)
        self.text_entry = tk.Entry(root)
        self.text_entry.pack(pady=5)
        
        # Number input field
        self.number_label = tk.Label(root, text='Number Input:')
        self.number_label.pack(pady=5)
        self.number_entry = tk.Spinbox(root, from_=0, to=100)
        self.number_entry.pack(pady=5)
        
        # Submit button
        self.submit_btn = tk.Button(root, text='Submit', command=self.submit)
        self.submit_btn.pack(pady=10)

    def upload_reductions(self):
        self.reduction_csv_paths = filedialog.askopenfilenames(title='Select Files')
        if self.reduction_csv_paths:
            messagebox.showinfo('Selected Files', '\n'.join(self.reduction_csv_paths))

    def upload_custom_models(self):
        self.custom_model_paths = filedialog.askopenfilenames(title='Select Files')
        if self.custom_model_paths:
            messagebox.showinfo('Selected Files', '\n'.join(self.custom_model_paths))

    def upload_base(self):
        self.base_csv_path = filedialog.askopenfilename(title='Select File')
        if self.base_csv_path:
            messagebox.showinfo('Selected File', self.base_csv_path)

    def submit(self):
        selected_options = []
        if self.check_var1.get():
            selected_options.append('Option 1')
        if self.check_var2.get():
            selected_options.append('Option 2')
        
        text_input = self.text_entry.get()
        number_input = self.number_entry.get()
        
        result = f'Selected Options: {", ".join(selected_options)}\n'
        result += f'Text Input: {text_input}\n'
        result += f'Number Input: {number_input}'
        
        messagebox.showinfo('Submitted Data', result)

if __name__ == '__main__':
    root = tk.Tk()
    ui = SimpleUI(root)
    root.mainloop()