import tkinter as tk
from tkinter import filedialog, messagebox, Frame
# from test_suite import Tester

class SimpleUI:
    def __init__(self, root: tk.Tk):
        self.base_csv_path = None
        self.reduction_csv_paths = None
        self.custom_model_paths = []
    
        root.title('Dataset Reduction Test Suite')

        # Set default window size
        root.geometry('450x450')
        
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
        default_models_frame=Frame(root, width=400, height=200)
        default_models_frame.pack()

        self.resnet_simple = tk.IntVar()
        self.resnet_34 = tk.IntVar()
        self.resnet_50 = tk.IntVar()
        self.efficient_net = tk.IntVar()
        self.mobile_net = tk.IntVar()

        self.resent_simple_btn = tk.Checkbutton(default_models_frame, text='Resnet (Simple)', variable=self.resnet_simple)
        self.resnet_34_btn = tk.Checkbutton(default_models_frame, text='Resnet 34', variable=self.resnet_34)
        self.resnet_50_btn = tk.Checkbutton(default_models_frame, text='Resnet 50', variable=self.resnet_50)
        self.efficient_net_btn = tk.Checkbutton(default_models_frame, text='EfficientNet', variable=self.efficient_net)
        self.mobile_net_btn = tk.Checkbutton(default_models_frame, text='MobileNet', variable=self.mobile_net)

        self.resent_simple_btn.grid(row=0, column=0)
        self.resnet_34_btn.grid(row=0, column=1)
        self.resnet_50_btn.grid(row=0, column=2)
        self.efficient_net_btn.grid(row=1, column=0)
        self.mobile_net_btn.grid(row=1, column=1)
        
        self.resnet_simple.set(True)
        self.resnet_34.set(True)
        self.resnet_50.set(True)
        self.efficient_net.set(True)
        self.mobile_net.set(True)
        
        # Text input field
        shape_frame=Frame(root, width=400, height=200)
        shape_frame.pack(fill='x', pady=10)
        self.shape_label = tk.Label(shape_frame, text='Input Shape (comma separated, no batch dim):')
        self.shape_input = tk.Entry(shape_frame)
        self.shape_label.pack(side='left', padx=25)
        self.shape_input.pack(side='right', padx=25)
        
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
        shape_input = self.shape_input.get()
        
        # ensure required fields are present
        required = {
            'Base csv path': self.base_csv_path,
            'Reduction csv path(s)': self.reduction_csv_paths,
            'Input shape': shape_input
        }
        missing = [key for key in required if not required[key]]
        if len(missing) > 0:
            messagebox.showinfo('Missing Fields', f'Missing required fields: {", ".join(missing)}')
            return
        
        # ensure input shape is correctly formatted
        if shape_input:
            input_clean = shape_input.replace(' ', '').replace('(', '').replace(')', '')
            dim_list = input_clean.split(',')
            dim_list_int = []
            for dim in dim_list:
                if len(dim) == 0: continue
                if not dim.isdecimal():
                    messagebox.showinfo('Bad Format', f'Incorrect format for Input Shape. Problem near: "{dim}"')
                    return
                dim_list_int.append(int(dim))

            input_shape = tuple(dim_list_int)
        else:
            input_shape = None

        selected_models = []
        if self.resnet_simple.get():
            selected_models.append('resnet_simple')
        if self.resnet_34.get():
            selected_models.append('resnet_34')
        if self.resnet_50.get():
            selected_models.append('resnet_50')
        if self.efficient_net.get():
            selected_models.append('efficient_net')
        if self.mobile_net.get():
            selected_models.append('mobile_net')

        number_input = self.number_entry.get()
        
        result = f'Selected Options: {", ".join(selected_models)}\n'
        result += f'Parsed Input Shape: {input_shape}\n'
        result += f'Number Input: {number_input}'
        
        messagebox.showinfo('Submitted Data', result)

if __name__ == '__main__':
    root = tk.Tk()
    ui = SimpleUI(root)
    root.mainloop()