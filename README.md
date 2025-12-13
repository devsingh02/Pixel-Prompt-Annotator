# ✨ Annotation Assistant

![Demo](demo.jpg)

## Overview
Annotation Assistant is a state-of-the-art **Vision-Language Object Detection** tool. It combines the power of **Qwen-VL (4B)** with a user-friendly interface to make labeled data creation effortless.

Unlike standard detection tools, this assistant is **conversational**. You can refine detections naturally (e.g., *"Also find the cup"*), and the AI intelligently merges new findings with existing ones.

## Key Features

### 🧠 **Intelligent Memory & Context**
The Assistant remembers what it has already found.
*   **No Amnesia**: Unlike basic wrappers, this tool feeds its own previous detections back into the context.
*   **Example**: If you say *"Find the laptop"* and then *"Find the remaining objects"*, it understands what "remaining" means because it knows the laptop is already detected.

### 🎯 **Smart Refinement Logic**
I implemented a custom **Weighted Merge Algorithm** to handle updates:
*   **Refinement**: If you draw a better box for `"shirt"` over an existing one (>80% overlap), it **replaces** the old one.
*   **Distinct Objects**: If you seek a second `"shirt"` elsewhere (low overlap), it **adds** it as a new object.
*   Result: NO duplicate ghost boxes, NO accidental deletions.

### 👁️ **Explainable AI (Reasoning)**
Don't just trust the box. The Assistant provides a **Reasoning Stream** explaining *why* it detected an object.
*   *Example*: "Detected silver laptop due to distinct Apple logo and metallic finish."

## How to Run

### ☁️ Option 1: Google Colab (Recommended for Free GPU)
1.  Open the `Colab_Runner.ipynb` file in Google Colab.
2.  Upload `app.py`, `utils.py`, and `requirements.txt` to the Colab files area.
3.  Add your **Ngrok Authtoken** in the designated cell.
4.  Run all cells. The app will launch via a public URL.

### 🤗 Option 2: Hugging Face Spaces (CPU/GPU)
1.  Create a new Space on Hugging Face.
2.  Select **Streamlit** as the SDK.
3.  Upload the files from this repository.
4.  The app will build and launch automatically.

### 💻 Option 3: Local System (Requires GPU)
1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/devsingh02/Pixel-Prompt-Annotator.git
    cd Pixel-Prompt-Annotator
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

---
*Built with Streamlit, Qwen-VL, and ❤️.*
