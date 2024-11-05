import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import albumentations as A
import os
import time
from functools import partial

class AugmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Image Augmentation Tool")
        
        # Store original image
        self.original_image = None
        self.current_image = None
        self.preview_cache = {}
        self.last_update_time = 0
        self.update_delay = 100 #ms
        self.pending_update = None
        
        # Params
        self.params = {
            # Geometric Transforms
            "rotate": {
                "active": False, 
                "prob": 1.0, 
                "range": 30,
                "min": -180, 
                "max": 180,
                "step": 1
            },
            "scale": {
                "active": False,
                "prob": 1.0,
                "range": 1.0,
                "min": 0.5,
                "max": 2.0,
                "step": 0.1
            },
            "shift_x": {
                "active": False,
                "prob": 1.0,
                "range": 0.1,
                "min": -0.3,
                "max": 0.3,
                "step": 0.05
            },
            "shift_y": { 
                "active": False,
                "prob": 1.0,
                "range": 0.1,
                "min": -0.3,
                "max": 0.3,
                "step": 0.05
            },
            "flip_horizontal": {
                "active": False, 
                "prob": 1.0
            },
            "flip_vertical": {
                "active": False, 
                "prob": 1.0
            },
            
            # Color Transforms
            "brightness": {
                "active": False, 
                "prob": 1.0, 
                "range": 0.2,
                "min": 0.0, 
                "max": 1.0,
                "step": 0.1
            },
            "contrast": {
                "active": False, 
                "prob": 1.0, 
                "range": 0.2,
                "min": 0.0, 
                "max": 1.0,
                "step": 0.1
            },
            "hue": {
                "active": False,
                "prob": 1.0,
                "range": 20,
                "min": 0,
                "max": 60,
                "step": 5
            },
            "saturation": {
                "active": False,
                "prob": 1.0,
                "range": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1
            },
            
            # Noise & Blur
            "blur": {
                "active": False, 
                "prob": 1.0, 
                "range": 3,
                "min": 3, 
                "max": 11,
                "step": 2
            },
            "gaussian_noise": {
                "active": False,
                "prob": 1.0,
                "range": 20,
                "min": 5,
                "max": 50,
                "step": 5
            },
            "motion_blur": {
                "active": False,
                "prob": 1.0,
                "range": 3,
                "min": 3,
                "max": 15,
                "step": 2
            },
            
            # Weather & Lighting
            "shadow": {
                "active": False,
                "prob": 1.0,
                "range": 0.5,
                "min": 0.3,
                "max": 0.7,
                "step": 0.1
            },
            "fog": {
                "active": False,
                "prob": 1.0,
                "range": 0.3,
                "min": 0.1,
                "max": 0.8,
                "step": 0.1
            }
        }

        # Create main frame with weight configuration
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Configure main frame weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        # Create preview frames with equal width columns
        self.preview_frame = ttk.Frame(self.main_frame)
        self.preview_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.columnconfigure(1, weight=1)
        
        self.original_preview_frame = ttk.LabelFrame(self.preview_frame, text="Original Image")
        self.original_preview_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        self.original_label = ttk.Label(self.original_preview_frame)
        self.original_label.pack(padx=5, pady=5)
        
        self.preview_preview_frame = ttk.LabelFrame(self.preview_frame, text="Augmented Image")
        self.preview_preview_frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        self.preview_label = ttk.Label(self.preview_preview_frame)
        self.preview_label.pack(padx=5, pady=5)
        
        self.load_btn = ttk.Button(self.main_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window inside canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Grid layout for canvas and scrollbar
        self.canvas.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5)
        self.scrollbar.grid(row=2, column=2, sticky="ns")
        
        # Configure weights
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.canvas.grid_configure(sticky="nsew")
        
        # Bind mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Create parameter controls inside scrollable frame
        self.create_parameter_controls()

        # Set maximum height for canvas
        self.canvas.update_idletasks()
        canvas_height = min(400, self.scrollable_frame.winfo_reqheight())
        self.canvas.configure(height=canvas_height)

        self.folder_frame = ttk.LabelFrame(self.main_frame, text="Batch Processing", padding="5")
        self.folder_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        input_frame = ttk.Frame(self.folder_frame)
        input_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(input_frame, text="Input Folder:").grid(row=0, column=0, padx=5)
        self.input_path_var = tk.StringVar()
        self.input_path_entry = ttk.Entry(input_frame, textvariable=self.input_path_var)
        self.input_path_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse", command=self.select_input_folder).grid(row=0, column=2, padx=5)
        
        output_frame = ttk.Frame(self.folder_frame)
        output_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(output_frame, text="Output Folder:").grid(row=0, column=0, padx=5)
        self.output_path_var = tk.StringVar()
        self.output_path_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        self.output_path_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="Browse", command=self.select_output_folder).grid(row=0, column=2, padx=5)
        
        aug_frame = ttk.Frame(self.folder_frame)
        aug_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Left side - Number of augmentations
        aug_num_frame = ttk.Frame(aug_frame)
        aug_num_frame.grid(row=0, column=0, sticky=(tk.W))
        
        ttk.Label(aug_num_frame, text="Augmentations per image:").grid(row=0, column=0, padx=5)
        self.num_aug_var = tk.StringVar(value="5")
        self.num_aug_entry = ttk.Entry(aug_num_frame, textvariable=self.num_aug_var, width=10)
        self.num_aug_entry.grid(row=0, column=1, padx=5)
        
        # Right side - Random mode checkbox
        aug_mode_frame = ttk.Frame(aug_frame)
        aug_mode_frame.grid(row=0, column=1, sticky=(tk.E))
        
        self.random_aug_var = tk.BooleanVar(value=False)
        self.random_aug_check = ttk.Checkbutton(
            aug_mode_frame, 
            text="Random augs per one copy of image", 
            variable=self.random_aug_var
        )
        self.random_aug_check.grid(row=0, column=0, padx=5)

        # Configure the frame weights to spread components
        aug_frame.columnconfigure(0, weight=1)
        aug_frame.columnconfigure(1, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.folder_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=4, column=0, columnspan=2, pady=10)

        # Add buttons
        self.generate_btn = ttk.Button(
            self.button_frame, 
            text="Generate Code", 
            command=self.generate_code
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            self.button_frame,
            text="Save Augmented Image",
            command=self.save_augmented_image
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.run_btn = ttk.Button(
            self.button_frame,
            text="Run Batch Augmentation",
            command=self.run_batch_augmentation
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Configure weights for folder frames
        input_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(1, weight=1)
        self.folder_frame.columnconfigure(0, weight=1)
        
    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        width = event.width - 4
        if width > 1:
            self.canvas.itemconfig(1, width=width)
                
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_path_var.set(folder)
            # Load first image from folder as preview
            image_files = [f for f in os.listdir(folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            if image_files:
                self.load_image(os.path.join(folder, image_files[0]))
                
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path_var.set(folder)

    def run_batch_augmentation(self):
        input_folder = self.input_path_var.get()
        output_folder = self.output_path_var.get()
        
        if not input_folder or not output_folder:
            tk.messagebox.showwarning("Warning", "Please select both input and output folders!")
            return
        
        try:
            num_aug = int(self.num_aug_var.get())
            if num_aug <= 0:
                raise ValueError
        except ValueError:
            tk.messagebox.showwarning("Warning", "Please enter a valid number of augmentations!")
            return

        os.makedirs(output_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if not image_files:
            tk.messagebox.showwarning("Warning", "No image files found in input folder!")
            return

        # Transform pipeline from current settings
        transforms = []
        for param_name, param_data in self.params.items():
            if param_data['active_var'].get():
                prob = param_data['prob_var'].get()
                
                if param_name == "rotate":
                    angle = param_data['range_var'].get()
                    transforms.append(A.Rotate(limit=(-angle, angle), p=prob))
                
                elif param_name == "scale":
                    scale = param_data['range_var'].get()
                    transforms.append(A.RandomScale(scale_limit=(scale-1, scale-1), p=prob))
                
                elif param_name == "shift_x":
                    value = param_data['range_var'].get()
                    transforms.append(A.ShiftScaleRotate(
                        shift_limit_x=(-value, value),
                        shift_limit_y=0,
                        scale_limit=0,
                        rotate_limit=0,
                        p=prob
                    ))
                
                elif param_name == "shift_y":
                    value = param_data['range_var'].get()
                    transforms.append(A.ShiftScaleRotate(
                        shift_limit_x=0,
                        shift_limit_y=(-value, value),
                        scale_limit=0,
                        rotate_limit=0,
                        p=prob
                    ))
                
                elif param_name == "flip_horizontal":
                    transforms.append(A.HorizontalFlip(p=prob))
                
                elif param_name == "flip_vertical":
                    transforms.append(A.VerticalFlip(p=prob))
                
                elif param_name == "brightness":
                    value = param_data['range_var'].get()
                    transforms.append(A.RandomBrightnessContrast(
                        brightness_limit=(-value, value),
                        contrast_limit=0,
                        p=prob
                    ))
                
                elif param_name == "contrast":
                    value = param_data['range_var'].get()
                    transforms.append(A.RandomBrightnessContrast(
                        brightness_limit=0,
                        contrast_limit=(-value, value),
                        p=prob
                    ))
                
                elif param_name == "hue":
                    value = param_data['range_var'].get()
                    transforms.append(A.HueSaturationValue(
                        hue_shift_limit=(-value, value),
                        sat_shift_limit=0,
                        val_shift_limit=0,
                        p=prob
                    ))
                
                elif param_name == "saturation":
                    value = param_data['range_var'].get()
                    transforms.append(A.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=(-int(value * 100), int(value * 100)),
                        val_shift_limit=0,
                        p=prob
                    ))
                
                elif param_name == "blur":
                    value = int(param_data['range_var'].get()) // 2 * 2 + 1
                    transforms.append(A.Blur(blur_limit=value, p=prob))
                
                elif param_name == "gaussian_noise":
                    value = param_data['range_var'].get()
                    transforms.append(A.GaussNoise(
                        var_limit=(0, value**2),
                        p=prob
                    ))
                
                elif param_name == "motion_blur":
                    value = int(param_data['range_var'].get()) // 2 * 2 + 1
                    transforms.append(A.MotionBlur(
                        blur_limit=value,
                        p=prob
                    ))
                
                elif param_name == "shadow":
                    value = param_data['range_var'].get()
                    transforms.append(A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        shadow_dimension=5,
                        p=prob
                    ))
                
                elif param_name == "fog":
                    value = param_data['range_var'].get()
                    transforms.append(A.RandomFog(
                        fog_coef_lower=max(0.1, value-0.1),
                        fog_coef_upper=min(1.0, value+0.1),
                        alpha_coef=0.08,
                        p=prob
                    ))

        transform = A.Compose(transforms)
        
        total_operations = len(image_files) * num_aug
        completed_operations = 0
        
        try:
            for filename in image_files:
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {filename}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if self.random_aug_var.get():
                    # Random mode: each image gets random combination of augmentations
                    transform_list = transforms.copy()
                    for i in range(num_aug):
                        current_transforms = []
                        for t in transform_list:
                            if np.random.random() < t.p:
                                current_transforms.append(t)
                        
                        if current_transforms:
                            current_transform = A.Compose(current_transforms)
                            augmented = current_transform(image=image)
                            augmented_image = augmented['image']
                        else:  # If no transforms were selected, use original image
                            augmented_image = image
                        
                        # Save augmented image
                        output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                        
                        # Update progress
                        completed_operations += 1
                        self.progress_var.set((completed_operations / total_operations) * 100)
                        self.root.update_idletasks()
                else:
                    # Original mode: apply all transforms with their probabilities
                    for i in range(num_aug):
                        augmented = transform(image=image)
                        augmented_image = augmented['image']
                        
                        # Save augmented image
                        output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                        output_path = os.path.join(output_folder, output_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                        
                        # Update progress
                        completed_operations += 1
                        self.progress_var.set((completed_operations / total_operations) * 100)
                        self.root.update_idletasks()
            
            tk.messagebox.showinfo("Success", "Batch augmentation completed successfully!")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error during batch processing: {str(e)}")
        
        finally:
            self.progress_var.set(0)
            
    def delayed_update(self):
        """Schedule a delayed update"""
        if self.pending_update:
            self.root.after_cancel(self.pending_update)
        self.pending_update = self.root.after(self.update_delay, self.update_preview)
    
    def create_parameter_controls(self):
        # Create frames for left and right columns inside scrollable frame
        left_frame = ttk.Frame(self.scrollable_frame)
        right_frame = ttk.Frame(self.scrollable_frame)
        
        left_frame.grid(row=0, column=0, padx=5, sticky="nsew")
        right_frame.grid(row=0, column=1, padx=5, sticky="nsew")
        
        # Configure column weights for scrollable frame
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.columnconfigure(1, weight=1)
        
        # Divide parameters into two groups
        params_list = list(self.params.items())
        mid_point = len(params_list) // 2
        left_params = params_list[:mid_point]
        right_params = params_list[mid_point:]
        
        # Create controls for both columns
        self.create_column_controls(left_frame, left_params)
        self.create_column_controls(right_frame, right_params)

        # Configure weights for frames
        left_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

    def create_column_controls(self, parent, params):
        for row, (param_name, param_data) in enumerate(params):
            # Parameter frame
            param_frame = ttk.LabelFrame(
                parent, 
                text=param_name.replace('_', ' ').title(), 
                padding="5"
            )
            param_frame.grid(row=row, column=0, pady=5, sticky=(tk.W, tk.E))
            
            # Checkbox for activation
            var = tk.BooleanVar(value=param_data['active'])
            self.params[param_name]['active_var'] = var
            check = ttk.Checkbutton(
                param_frame, 
                text="Active", 
                variable=var,
                command=lambda pn=param_name: self.on_scale_change(pn)
            )
            check.grid(row=0, column=0, padx=5)
            
            # Probability controls frame
            prob_frame = ttk.Frame(param_frame)
            prob_frame.grid(row=0, column=1, columnspan=2, padx=5, sticky=(tk.W, tk.E))
            
            ttk.Label(prob_frame, text="Probability:").grid(row=0, column=0, padx=5)
            
            # Probability scale
            prob_var = tk.DoubleVar(value=param_data['prob'])
            self.params[param_name]['prob_var'] = prob_var
            prob_scale = ttk.Scale(
                prob_frame, 
                from_=0.0, 
                to=1.0, 
                orient=tk.HORIZONTAL, 
                variable=prob_var,
                command=lambda x, pn=param_name: self.on_scale_change(pn)
            )
            prob_scale.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
            
            # Clickable probability value entry
            prob_entry = ttk.Entry(prob_frame, width=6)
            prob_entry.grid(row=0, column=2, padx=5)
            prob_entry.insert(0, f"{param_data['prob']:.2f}")
            prob_entry.bind('<Return>', lambda e, pn=param_name, entry=prob_entry: 
                self.on_entry_change(pn, entry, 'prob'))
            prob_entry.bind('<FocusOut>', lambda e, pn=param_name, entry=prob_entry: 
                self.on_entry_change(pn, entry, 'prob'))
            self.params[param_name]['prob_entry'] = prob_entry
            
            # Add range control if parameter has range settings
            if 'range' in param_data:
                self.add_range_control(param_frame, param_name, row=1)


    def add_range_control(self, parent, param_name, row):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, pady=5)
        
        ttk.Label(frame, text="Range:").grid(row=0, column=0, padx=5)
        
        # Range scale
        range_var = tk.DoubleVar(value=self.params[param_name]['range'])
        self.params[param_name]['range_var'] = range_var
        
        scale = ttk.Scale(
            frame,
            from_=self.params[param_name]['min'],
            to=self.params[param_name]['max'],
            orient=tk.HORIZONTAL,
            variable=range_var,
            command=lambda x, pn=param_name: self.on_scale_change(pn)
        )
        scale.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Clickable range value entry
        range_entry = ttk.Entry(frame, width=6)
        range_entry.grid(row=0, column=2, padx=5)
        range_entry.insert(0, str(self.params[param_name]['range']))
        range_entry.bind('<Return>', lambda e, pn=param_name, entry=range_entry: 
            self.on_entry_change(pn, entry, 'range'))
        range_entry.bind('<FocusOut>', lambda e, pn=param_name, entry=range_entry: 
            self.on_entry_change(pn, entry, 'range'))
        self.params[param_name]['range_entry'] = range_entry
    
    def on_entry_change(self, param_name, entry, entry_type):
        try:
            value = float(entry.get())
            if entry_type == 'prob':
                value = max(0.0, min(1.0, value))
                self.params[param_name]['prob_var'].set(value)
                entry.delete(0, tk.END)
                entry.insert(0, f"{value:.2f}")
            else:  # range
                min_val = self.params[param_name]['min']
                max_val = self.params[param_name]['max']
                value = max(min_val, min(max_val, value))
                self.params[param_name]['range_var'].set(value)
                entry.delete(0, tk.END)
                entry.insert(0, f"{value:.2f}")
            
            # delayed update for noise parameters
            if param_name in ['gaussian_noise', 'speckle_noise']:
                self.delayed_update()
            else:
                self.update_preview()
        except ValueError:
            # Reset to current value if invalid input
            if entry_type == 'prob':
                entry.delete(0, tk.END)
                entry.insert(0, f"{self.params[param_name]['prob_var'].get():.2f}")
            else:
                entry.delete(0, tk.END)
                entry.insert(0, f"{self.params[param_name]['range_var'].get():.2f}")


    def on_scale_change(self, param_name):
        # Update entries when scales change
        if 'prob_entry' in self.params[param_name]:
            prob = self.params[param_name]['prob_var'].get()
            self.params[param_name]['prob_entry'].delete(0, tk.END)
            self.params[param_name]['prob_entry'].insert(0, f"{prob:.2f}")
        
        if 'range_entry' in self.params[param_name]:
            range_val = self.params[param_name]['range_var'].get()
            if param_name == 'blur':
                range_val = int(range_val) // 2 * 2 + 1
            self.params[param_name]['range_entry'].delete(0, tk.END)
            self.params[param_name]['range_entry'].insert(0, f"{range_val:.2f}")
        
        # Use delayed update for noise parameters
        if param_name in ['gaussian_noise', 'speckle_noise']:
            self.delayed_update()
        else:
            self.update_preview()
    
    @staticmethod
    def resize_image(image, max_size=300):
        """Efficiently resize image maintaining aspect ratio"""
        height, width = image.shape[:2]
        scale = min(max_size/width, max_size/height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def load_image(self, path=None):
        """
        Load image either from file dialog or from given path
        """
        if path is None:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )
        else:
            file_path = path
            
        if file_path:
            # Load and cache preview-sized image
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.preview_size_image = self.resize_image(self.original_image)
            self.original_preview = Image.fromarray(self.preview_size_image)
            
            # Clear any cached transforms
            self.preview_cache.clear()
            
            self.update_preview()

    def update_preview(self):
        if self.original_image is None:
            return
            
        current_time = time.time()
        if current_time - self.last_update_time < 0.05:  #ms
            self.delayed_update()
            return

        transforms = []
        for param_name, param_data in self.params.items():
            if param_data['active_var'].get():
                value = param_data['range_var'].get() if 'range_var' in param_data else None
                
                if param_name == "rotate":
                    transforms.append(A.Rotate(limit=(value, value), p=1.0))
                elif param_name == "scale":
                    transforms.append(A.RandomScale(scale_limit=(value-1, value-1), p=1.0))
                elif param_name == "shift_x":
                    transforms.append(A.ShiftScaleRotate(
                        shift_limit_x=(value, value),
                        shift_limit_y=0,
                        scale_limit=0,
                        rotate_limit=0,
                        p=1.0
                    ))
                elif param_name == "shift_y":
                    transforms.append(A.ShiftScaleRotate(
                        shift_limit_x=0,
                        shift_limit_y=(value, value),
                        scale_limit=0,
                        rotate_limit=0,
                        p=1.0
                    ))
                elif param_name == "flip_horizontal":
                    transforms.append(A.HorizontalFlip(p=1.0))
                elif param_name == "flip_vertical":
                    transforms.append(A.VerticalFlip(p=1.0))
                elif param_name == "brightness":
                    transforms.append(A.RandomBrightnessContrast(
                        brightness_limit=(value, value),
                        contrast_limit=0,
                        p=1.0
                    ))
                elif param_name == "contrast":
                    transforms.append(A.RandomBrightnessContrast(
                        brightness_limit=0,
                        contrast_limit=(value, value),
                        p=1.0
                    ))
                elif param_name == "hue":
                    transforms.append(A.HueSaturationValue(
                        hue_shift_limit=(value, value),
                        sat_shift_limit=0,
                        val_shift_limit=0,
                        p=1.0
                    ))
                elif param_name == "saturation":
                    transforms.append(A.HueSaturationValue(
                        hue_shift_limit=0,
                        sat_shift_limit=(int(value * 100), int(value * 100)),
                        val_shift_limit=0,
                        p=1.0
                    ))
                elif param_name == "blur":
                    blur_value = int(value) // 2 * 2 + 1
                    transforms.append(A.Blur(blur_limit=blur_value, p=1.0))
                elif param_name == "gaussian_noise":
                    transforms.append(A.GaussNoise(
                        var_limit=(value**2, value**2),
                        p=1.0
                    ))
                elif param_name == "motion_blur":
                    blur_value = int(value) // 2 * 2 + 1
                    transforms.append(A.MotionBlur(blur_limit=blur_value, p=1.0))
                elif param_name == "shadow":
                    transforms.append(A.RandomShadow(
                        shadow_roi=(0, 0, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=1,
                        shadow_dimension=5,
                        p=1.0
                    ))
                elif param_name == "fog":
                    transforms.append(A.RandomFog(
                        fog_coef_lower=value,
                        fog_coef_upper=value,
                        alpha_coef=0.08,
                        p=1.0
                    ))

        transform = A.Compose(transforms)
        
        try:
            # Use cached preview size for faster processing
            if not hasattr(self, 'preview_size_image'):
                self.preview_size_image = self.resize_image(self.original_image)
            
            # Apply transforms to preview-sized image
            augmented = transform(image=self.preview_size_image)
            self.current_image = augmented['image']
            
            # Always show both original and augmented images
            self.show_preview(self.preview_size_image, self.original_label)
            self.show_preview(self.current_image, self.preview_label)
            self.last_update_time = current_time
            
        except Exception as e:
            print(f"Error applying transforms: {str(e)}")
    
    def show_preview(self, image, label):
        """Efficiently convert and display image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo
    
    def save_augmented_image(self):
        if self.original_image is None or not self.params:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            # Apply transforms to full-size image for saving
            transforms = []
            transform = A.Compose(transforms)
            
            try:
                augmented = transform(image=self.original_image)
                cv2.imwrite(file_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error saving augmented image: {str(e)}")
    def generate_code(self):
        if self.original_image is None:
            tk.messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        code = """import albumentations as A
import cv2
import os

def create_augmentation_pipeline():
    transform = A.Compose([
"""
        
        # Add selected transforms with ranges for actual augmentation
        for param_name, param_data in self.params.items():
            if param_data['active_var'].get():
                prob = param_data['prob_var'].get()
                
                if param_name == "rotate":
                    angle = param_data['range_var'].get()
                    code += f"        A.Rotate(limit=({-angle}, {angle}), p={prob:.2f}),\n"
                elif param_name == "flip_horizontal":
                    code += f"        A.HorizontalFlip(p={prob:.2f}),\n"
                elif param_name == "flip_vertical":
                    code += f"        A.VerticalFlip(p={prob:.2f}),\n"
                elif param_name == "brightness":
                    bright_value = param_data['range_var'].get()
                    code += f"        A.RandomBrightnessContrast(brightness_limit=({-bright_value}, {bright_value}), contrast_limit=0, p={prob:.2f}),\n"
                elif param_name == "contrast":
                    contrast_value = param_data['range_var'].get()
                    code += f"        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=({-contrast_value}, {contrast_value}), p={prob:.2f}),\n"
                elif param_name == "blur":
                    blur_value = int(param_data['range_var'].get()) // 2 * 2 + 1
                    code += f"        A.Blur(blur_limit=({blur_value}, {blur_value}), p={prob:.2f}),\n"
                elif param_name == "noise":
                    noise_std = param_data['range_var'].get()
                    code += f"        A.GaussNoise(var_limit=({noise_std**2}, {noise_std**2}), p={prob:.2f}),\n"

        code += """    ])
    return transform

def augment_images(input_folder, output_folder, num_augmentations=5):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            # Read image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate augmentations
            for i in range(num_augmentations):
                # Apply transforms
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                # Save augmented image
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Replace these with your actual input and output folder paths
    INPUT_FOLDER = "input_images"
    OUTPUT_FOLDER = "augmented_images"
    
    augment_images(INPUT_FOLDER, OUTPUT_FOLDER)
"""
        
        code_window = tk.Toplevel(self.root)
        code_window.title("Generated Code")
        
        text_widget = tk.Text(code_window, wrap=tk.NONE)
        scrollbar = ttk.Scrollbar(code_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(text_widget.get("1.0", tk.END))
        
        copy_btn = ttk.Button(code_window, text="Copy to Clipboard", command=copy_to_clipboard)
        copy_btn.pack(pady=5)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert("1.0", code)

def main():
    root = tk.Tk()
    app = AugmentationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()