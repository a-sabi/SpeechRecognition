import tkinter as tk
from tkinter import filedialog
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import whisper
import customtkinter

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

class SpeechRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Speech Recognition App")

        # Set up styles
        self.set_styles()

        self.whisper_models = ["small", "medium", "large-v2"]
        self.transformer_models = ["MMS-1B:FL102", "MMS-1B:L1107", "MMS-1B-all"]
        self.languages = ["English", "Russian", "German", "French"]
        self.language_codes = {"English": "eng", "Russian": "rus", "German": "deu", "French": "fra"}

        self.model_type = tk.StringVar(master)
        self.model_type.set("Whisper")
        self.model_select = tk.StringVar(master)
        self.model_select.set(self.whisper_models[0])
        self.language_select = tk.StringVar(master)
        self.language_select.set(self.languages[0])

        self.create_widgets()

    def set_styles(self):
        self.master.configure(background='#000000')
        self.master.option_add('*Font', 'Arial 10')

    def create_widgets(self):
        # Model Type
        self.model_type_label = tk.Label(self.master, text="Select Model Type:", bg='#000000', fg='white')
        self.model_type_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.model_type_menu = tk.OptionMenu(self.master, self.model_type, "Whisper", "Transformer",
                                             command=self.update_model_menu)
        self.model_type_menu.config(bg='#2196F3', fg='white', activebackground='#64B5F6', activeforeground='white',
                                    bd=4, relief=tk.RAISED)
        self.model_type_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.model_type_menu['menu'].config(bg='#2196F3', fg='white', activebackground='#64B5F6',
                                            activeforeground='white')

        # Model Select
        self.model_select_label = tk.Label(self.master, text="Select Model:", bg='#000000', fg='white')
        self.model_select_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.model_select_menu = tk.OptionMenu(self.master, self.model_select, *self.whisper_models)
        self.model_select_menu.config(bg='#2196F3', fg='white', activebackground='#64B5F6', activeforeground='white',
                                      bd=4, relief=tk.RAISED)
        self.model_select_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.model_select_menu['menu'].config(bg='#2196F3', fg='white', activebackground='#64B5F6',
                                              activeforeground='white')

        # Language Select
        self.language_select_label = tk.Label(self.master, text="Select Language:", bg='#000000', fg='white')
        self.language_select_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.language_select_menu = tk.OptionMenu(self.master, self.language_select, *self.languages)
        self.language_select_menu.config(bg='#2196F3', fg='white', activebackground='#64B5F6', activeforeground='white',
                                          bd=4, relief=tk.RAISED)
        self.language_select_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.language_select_menu['menu'].config(bg='#2196F3', fg='white', activebackground='#64B5F6',
                                                 activeforeground='white')

        # Buttons
        self.browse_button = tk.Button(self.master, text="Browse Audio File", command=self.browse_file, bg='#2196F3', fg='white',
                                       bd=4, relief=tk.RAISED, padx=10)
        self.browse_button.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.recognize_button = tk.Button(self.master, text="Recognize Speech", command=self.recognize_speech, bg='#2196F3', fg='white',
                                          bd=4, relief=tk.RAISED, padx=10)
        self.recognize_button.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.save_button = tk.Button(self.master, text="Save Text", command=self.save_text, bg='#2196F3', fg='white',
                                     bd=4, relief=tk.RAISED, padx=10)
        self.save_button.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # Output Text
        self.output_text = tk.Text(self.master, height=10, width=50)
        self.output_text.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

    def update_model_menu(self, *args):
        if self.model_type.get() == "Whisper":
            self.model_select_menu['menu'].delete(0, 'end')
            for model in self.whisper_models:
                self.model_select_menu['menu'].add_command(label=model, command=tk._setit(self.model_select, model))
        else:
            self.model_select_menu['menu'].delete(0, 'end')
            for model in self.transformer_models:
                self.model_select_menu['menu'].add_command(label=model, command=tk._setit(self.model_select, model))

    def browse_file(self):
        self.audio_file_path = filedialog.askopenfilename(initialdir="/", title="Select Audio File",
                                                          filetypes=(("WAV files", "*.wav"), ("all files", "*.*")))
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Selected file: {self.audio_file_path}")

    def recognize_speech(self):
        if self.model_type.get() == "Whisper":
            model = whisper.load_model(self.model_select.get())
            result = model.transcribe(self.audio_file_path)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result["text"])
        else:
            language_code = self.language_codes[self.language_select.get()]
            model_id = f"facebook/{self.model_select.get().replace(':', '-')}"
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)

            processor.tokenizer.set_target_lang(language_code)
            model.load_adapter(language_code)

            waveform, sample_rate = torchaudio.load(self.audio_file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            inputs = processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt")

            with torch.no_grad():
                outputs = model(input_values=inputs.input_values, attention_mask=inputs.attention_mask).logits

            predicted_ids = torch.argmax(outputs, dim=-1)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, transcription[0])

    def save_text(self):
        if hasattr(self, 'output_text'):
            text_to_save = self.output_text.get(1.0, tk.END)
            file_path = filedialog.asksaveasfilename(defaultextension=".txt")
            with open(file_path, 'w') as f:
                f.write(text_to_save)

root = tk.Tk()
app = SpeechRecognitionApp(root)
root.mainloop()
