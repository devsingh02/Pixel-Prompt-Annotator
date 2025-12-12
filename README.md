# ✨ Annotation Assistant

![Demo](demo_thumb.png)

## Overview
Annotation Assistant is a state-of-the-art **Vision-Language Object Detection** tool. It combines the power of **Qwen-VL (4B)** with a premium, user-friendly interface to make labeled data creation effortless.

Unlike standard detection tools, this assistant is **conversational**. You can refine detections naturally (e.g., *"Also find the cup"*), and the AI intelligently merges new findings with existing ones.

## Key Features

### 🧠 **Intelligent Memory & Context**
The Assistant remembers what it has already found.
*   **No Amnesia**: Unlike basic wrappers, this tool feeds its own previous detections back into the context.
*   **Example**: If you say *"Find the laptop"* and then *"Find the remaining objects"*, it understands what "remaining" means because it knows the laptop is already detected.

### 🎯 **Smart Refinement Logic**
We implemented a custom **Weighted Merge Algorithm** to handle updates:
*   **Refinement**: If you draw a better box for `"shirt"` over an existing one (>80% overlap), it **replaces** the old one.
*   **distinct Objects**: If you seek a second `"shirt"` elsewhere (low overlap), it **adds** it as a new object.
*   Result: NO duplicate ghost boxes, NO accidental deletions.

### 👁️ **Explainable AI (Reasoning)**
Don't just trust the box. The Assistant provides a **Reasoning Stream** explaining *why* it detected an object.
*   *Example*: "Detected silver laptop due to distinct Apple logo and metallic finish."

### 🎨 **Premium "Hero" Interface**
*   **Single-Column Layout**: Your image takes center stage.
*   **Dynamic Resizing**: Use the slider to scale the view from 300px to 1500px without losing layout structure.
*   **Visuals**: Deep Space gradient theme, glassmorphism metrics, and auto-centering.

## Quick Start
1.  **Upload**: Drag & Drop your image into the central hub.
2.  **Prompt**: Type what you're looking for (e.g., *"Find all branded items"*).
3.  **Refine**: Chat with the AI to fix mistakes or add more items.
4.  **Download**: Export your data as **COCO JSON** or download a **ZIP of cropped images**.

---
*Built with Streamlit, Qwen-VL, and ❤️.*
