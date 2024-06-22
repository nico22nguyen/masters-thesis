import tkinter as tk
from tkinter import filedialog, messagebox, Frame
from test_suite import Tester
from model_garden import MODEL

class SimpleUI:
    def __init__(self, root: tk.Tk):
        self.base_csv_path = None
        self.reduction_csv_paths = None
        self.custom_model_paths = None
    
        root.title('Dataset Reduction Test Suite')

        # Set default window size
        root.geometry('600x600')
        
        # Base csv upload button
        base_csv_frame=Frame(root)
        base_csv_frame.pack(fill='x', pady=(10, 0))

        self.base_upload_label = tk.Label(base_csv_frame, text='Base dataset (.csv):')
        self.base_upload_btn = tk.Button(base_csv_frame, text='Upload File', command=self.upload_base)
        self.base_required_text = tk.StringVar(value='[required]')
        self.base_required_value = tk.Label(base_csv_frame, textvariable=self.base_required_text, fg='red')

        self.base_upload_label.pack(side='left', padx=(25, 5))
        self.base_upload_btn.pack(side='right', padx=(5, 25))
        self.base_required_value.pack(side='left')
        
        self.base_display_frame=Frame(root)
        self.base_display_frame.pack(fill='x')

        self.base_display_var = tk.StringVar()
        self.base_display_label = tk.Label(self.base_display_frame, textvariable=self.base_display_var) # dont pack until user uploads a csv

        # Index csv upload button
        reduction_csv_frame=Frame(root)
        reduction_csv_frame.pack(fill='x', pady=(10, 0))

        self.reduction_upload_label = tk.Label(reduction_csv_frame, text='Reduction indices (.csv):')
        self.reduction_upload_btn = tk.Button(reduction_csv_frame, text='Upload File(s)', command=self.upload_reductions)
        self.reduction_required_text = tk.StringVar(value='[required]')
        self.reduction_required_tag = tk.Label(reduction_csv_frame, textvariable=self.reduction_required_text, fg='red')

        self.reduction_upload_label.pack(side='left', padx=(25, 5))
        self.reduction_upload_btn.pack(side='right', padx=(5, 25))
        self.reduction_required_tag.pack(side='left') 

        self.reduction_list_frame=Frame(root)
        self.reduction_list_frame.pack(fill='x')
        
        # Custom models upload button
        custom_models_frame=Frame(root)
        custom_models_frame.pack(fill='x', pady=(10, 0))

        self.model_upload_label = tk.Label(custom_models_frame, text='Custom Models (.keras, .pt):')
        self.custom_optional_text = tk.StringVar(value='[optional]')
        self.custom_optional_tag = tk.Label(custom_models_frame, textvariable=self.custom_optional_text)
        self.model_upload_btn = tk.Button(custom_models_frame, text='Upload File(s)', command=self.upload_custom_models)

        self.model_upload_label.pack(side='left', padx=(25, 5))
        self.model_upload_btn.pack(side='right', padx=(5, 25))
        self.custom_optional_tag.pack(side='left') 

        self.custom_list_frame=Frame(root)
        self.custom_list_frame.pack(fill='x')
        
        # Checkboxes
        default_models_label_frame=Frame(root)
        default_models_label_frame.pack(fill='x')
        self.default_models_label = tk.Label(default_models_label_frame, text='Default Models:')
        self.default_models_label.pack(side='left', padx=25, pady=(10, 0))

        default_models_frame=Frame(root)
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
        shape_frame=Frame(root)
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
        new_paths = filedialog.askopenfilenames(title='Select Files')
        if new_paths:
            # validate paths
            for path in new_paths:
                if path.split('.')[-1] != 'csv':
                    self.reduction_required_tag.pack(side='left')
                    self.reduction_required_text.set('Previous upload failed: All files must be .csv.')
                    return
            
            for label in self.reduction_list_frame.winfo_children():
                label.destroy()
            labels = [tk.Label(self.reduction_list_frame, text=f'- {path}') for path in new_paths]
            for label in labels: label.pack(side='top', padx=(50, 0), anchor='w')
            self.reduction_required_tag.pack_forget()
        else:
            self.reduction_required_text.set('[required]')
            self.reduction_required_tag.pack(side='left')
            for label in self.reduction_list_frame.winfo_children():
                label.destroy()
        
        self.reduction_csv_paths = new_paths

    def upload_custom_models(self):
        new_paths = filedialog.askopenfilenames(title='Select Files')
        if new_paths:
            # validate paths
            for path in new_paths:
                new_paths = list(set(new_paths))
                file_extension = path.split('.')[-1]
                if file_extension != 'keras' and file_extension != 'pt':
                    self.custom_optional_tag.pack(side='left')
                    self.custom_optional_tag.config(fg='red')
                    self.custom_optional_text.set('Previous upload failed: All files must be .keras or .pt.')
                    return

            for label in self.custom_list_frame.winfo_children():
                label.destroy()
            labels = [tk.Label(self.custom_list_frame, text=f'- {path}') for path in new_paths]
            for label in labels: label.pack(side='top', padx=(50, 0), anchor='w')
            self.custom_optional_tag.pack_forget()
        else:
            self.custom_optional_text.set('[optional]')
            self.custom_optional_tag.config(fg='black')
            self.custom_optional_tag.pack(side='left')
            for label in self.custom_list_frame.winfo_children():
                label.destroy()
        
        self.custom_model_paths = new_paths

    def upload_base(self):
        new_path = filedialog.askopenfilename(title='Select File')
        if new_path:
            if new_path.split('.')[-1] != 'csv':
                self.base_required_text.set('Previous upload failed: File must be .csv.')
                self.base_required_value.pack(side='left')
                return

            self.base_required_value.pack_forget()
            self.base_display_var.set(f'- {new_path}')
            self.base_display_label.pack(side='top', padx=(50, 0), anchor='w')
        else:
            self.base_display_label.pack_forget()
            self.base_required_text.set('[required]')
            self.base_required_value.pack(side='left')

        self.base_csv_path = new_path

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

        selected_models = []
        if self.resnet_simple.get():
            selected_models.append(MODEL.RESNET_SIMPLE)
        if self.resnet_34.get():
            selected_models.append(MODEL.RESNET_34)
        if self.resnet_50.get():
            selected_models.append(MODEL.RESNET_50)
        if self.efficient_net.get():
            selected_models.append(MODEL.EFFICIENT_NET)
        if self.mobile_net.get():
            selected_models.append(MODEL.MOBILE_NET)

        number_input = self.number_entry.get()
        
        print('initializing test suite...')
        test_suite = Tester(self.base_csv_path, input_shape, self.custom_model_paths, selected_models)
        print('success')
        result = f'Parsed Input Shape: {input_shape}\n'
        result += f'Number Input: {number_input}'
        
        messagebox.showinfo('Submitted Data', result)

if __name__ == '__main__':
    root = tk.Tk()
    ui = SimpleUI(root)
    root.mainloop()