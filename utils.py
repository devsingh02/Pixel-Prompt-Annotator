import os
import torch
import numpy as np
import json
import time
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForVision2Seq
import streamlit as st
import re

# Constants
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct" 

@st.cache_resource
def load_model():
    """
    Loads the Qwen-VL model and processor.
    """
    print(f"Loading model: {MODEL_ID}...")
    try:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # Use bfloat16 for CPU to save memory (4B params * 4 bytes is too big for 16GB)
        torch_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        print(f"Using device: {device_type}, dtype: {torch_dtype}")

        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
    except Exception as e:
        print(f"Error loading {MODEL_ID}: {e}")
        st.error(f"Could not load model {MODEL_ID}. Error: {e}")
        return None, None
        
    return processor, model

def get_bounding_boxes(image: Image.Image, prompt: str, history: list, processor, model):
    """
    Generates bounding boxes based on the image, prompt, and conversation history.
    """
    start_time = time.time()
    
    if model is None or processor is None:
        return [], history, "Model not loaded.", {}
        
    # Construct conversation
    messages = []
    
    # Context
    context_text = ""
    if history:
        context_text = "History:\n"
        for msg in history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_text += f"{role}: {msg['content']}\n"
        context_text += "\n"

    # Enhanced Prompt: JSON Focused With Reasoning
    final_prompt = f"{context_text}User Request: {prompt}\n\nTask: Detect objects mentioned in the User Request.\nConstraint: Return the result ONLY as a JSON object with a key 'objects'.\nEach object in the list should have 'label', 'bbox' [x1, y1, x2, y2] (common normalized coordinates 0-1000), AND 'reasoning' (a brief string explaining why this object matches).\nExample: {{'objects': [{{'label': 'cat', 'bbox': [100, 200, 500, 600], 'reasoning': 'Detected distinct feline features and whiskers.'}}]}}\nIf no objects are found, return {{'objects': []}}."

    messages = [
        {
            "role": "system", 
            "content": "You are a precise object detection assistant. Return JSON with 'objects' list containing 'label', 'bbox' [x1, y1, x2, y2] (common normalized coordinates 0-1000), and 'reasoning'."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_prompt}
            ]
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate (Measured)
        generate_start = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generate_end = time.time()
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
    except Exception as e:
        print(f"Inference Error: {e}")
        output_text = f"Error: {e}"
        generate_end = time.time()

    # Update history
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": output_text})
    
    # Parse detections
    detections = parse_qwen_output(output_text, image.width, image.height)
    
    # Filter
    filtered_detections = []
    total_area = image.width * image.height
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        box_area = (x2 - x1) * (y2 - y1)
        coverage = box_area / total_area
        
        is_suspicious_coverage = coverage > 0.95 
        is_whole_request = any(w in prompt.lower() for w in ["image", "picture", "photo", "background", "everything"])
        
        if is_suspicious_coverage and not is_whole_request:
            continue
        
        filtered_detections.append(det)
    
    # Metrics
    end_time = time.time()
    total_time = end_time - start_time
    inference_time = generate_end - generate_start
    
    metrics = {
        "total_time": round(total_time, 2),
        "inference_time": round(inference_time, 2),
        "token_count": len(generated_ids[0]) if 'generated_ids' in locals() else 0
    }
    
    return filtered_detections, history, output_text, metrics

def smart_merge_detections(existing_detections, new_detections):
    """
    Merges new detections with existing ones.
    Strategy: SIMPLE OVERLAP ONLY.
    If IoU > 0.8 -> Assume duplicate/refinement -> Replace.
    Else -> Keep.
    """
    merged_list = existing_detections.copy()
    
    for new_det in new_detections:
        new_box = new_det['box']
        indices_to_remove = []
        
        for i, old_det in enumerate(merged_list):
            old_box = old_det['box']
            iou = calculate_iou(new_box, old_box)
            
            # Simple threshold check
            if iou > 0.8:
                indices_to_remove.append(i)
        
        for idx in sorted(indices_to_remove, reverse=True):
            merged_list.pop(idx)
            
        merged_list.append(new_det)
        
    return merged_list

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def parse_qwen_output(text, width, height):
    """
    Parses Qwen-VL output, prioritizing JSON with reasoning.
    """
    detections = []
    
    # 1. Try JSON Parsing (Primary Strategy)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            data = json.loads(json_str)
            
            if 'objects' in data and isinstance(data['objects'], list):
                for obj in data['objects']:
                    x1, y1, x2, y2 = obj['bbox']
                    label = obj.get('label', 'Object')
                    reasoning = obj.get('reasoning', 'No reasoning provided')
                    
                    real_x1 = (x1 / 1000) * width
                    real_y1 = (y1 / 1000) * height
                    real_x2 = (x2 / 1000) * width
                    real_y2 = (y2 / 1000) * height
                    
                    detections.append({
                        "label": label,
                        "box": [real_x1, real_y1, real_x2, real_y2],
                        "score": 1.0,
                        "reasoning": reasoning
                    })
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        pass
            
    # 2. Fallback to Standard Tags
    if not detections:
        pattern_standard = r"<\|box_start\|>(\d+),(\d+),(\d+),(\d+)<\|box_end\|>(?:<\|object_start\|>(.*?)<\|object_end\|>)?"
        matches_standard = list(re.finditer(pattern_standard, text))
        for match in matches_standard:
            c1, c2, c3, c4 = map(int, match.groups()[:4])
            label = match.group(5) if match.group(5) else "Object"
            y1 = (c1 / 1000) * height
            x1 = (c2 / 1000) * width
            y2 = (c3 / 1000) * height
            x2 = (c4 / 1000) * width
            detections.append({
                "label": label,
                "box": [x1, y1, x2, y2],
                "score": 1.0,
                "reasoning": "Legacy detection mode"
            })

    return detections

def create_crops_zip(image: Image.Image, detections: list):
    """
    Creates a ZIP file containing cropped images of all detections.
    """
    zip_buffer = io.BytesIO()
    
    # Ensure distinct filenames
    counts = {}
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, det in enumerate(detections):
            label = det.get('label', 'object').replace(" ", "_").lower()
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] += 1
                label = f"{label}_{counts[label]}"
                
            x1, y1, x2, y2 = map(int, det['box'])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crop_buffer = io.BytesIO()
                crop.save(crop_buffer, format="JPEG")
                zip_file.writestr(f"{label}.jpg", crop_buffer.getvalue())
                
    zip_buffer.seek(0)
    return zip_buffer

def process_vision_info(messages):
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info(messages)
    except ImportError:
        images = []
        for msg in messages:
            for item in msg["content"]:
                if item["type"] == "image":
                    images.append(item["image"])
        return images, None

def draw_boxes(image: Image.Image, detections: list):
    """
    Draws bounding boxes with dynamic font scaling.
    """
    draw = ImageDraw.Draw(image)
    
    # Dynamic Scaling (UPDATED FOR BETTER VISIBILITY)
    min_dim = min(image.width, image.height)
    scaled_font_size = max(20, int(min_dim * 0.035)) 
    scaled_line_width = max(4, int(min_dim * 0.006))
    
    font = None
    try:
        # Search paths for fonts (Linux/Windows)
        font_paths = [
            # Windows
            "arial.ttf", "calibri.ttf", "seguiemj.ttf",
            # Linux (Standard)
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "DejaVuSans.ttf"
        ]
        
        for fn in font_paths:
            try:
                font = ImageFont.truetype(fn, scaled_font_size)
                print(f"Loaded font: {fn}")
                break
            except Exception as e:
                continue
    except:
        pass
        
    if font is None:
        try:
            print("Fallback to default font (Warning: Text may be tiny)")
            font = ImageFont.load_default()
        except:
             pass

    palette = [
        "#FF00FF", "#00FFFF", "#FF0000", "#00FF00", 
        "#FFFF00", "#FFA500", "#800080", "#008080"
    ]
    
    def get_color(text):
        if not text: return palette[0]
        idx = sum(ord(c) for c in text) % len(palette)
        return palette[idx]

    for det in detections:
        box = det['box']
        label = det.get('label', 'Object')
        score_val = det.get('score', 1.0)
        display_text = f"{label} {int(score_val*100)}%"
        
        color = get_color(label)
        
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=scaled_line_width)
        
        # Text box
        if font:
            text_bbox = draw.textbbox((x1, y1), display_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            label_y = y1 - text_height - (scaled_line_width * 2)
            if label_y < 0: label_y = y1
            
            draw.rectangle(
                [x1, label_y, x1 + text_width + (scaled_line_width * 4), label_y + text_height + (scaled_line_width * 2)], 
                fill=color
            )
            draw.text((x1 + (scaled_line_width), label_y), display_text, fill="black", font=font)
        
    return image

def convert_to_coco(detections, image_size=(1000, 1000), filename="image.jpg"):
    """
    Converts detections to full Standard COCO JSON format.
    """
    width, height = image_size
    
    # 1. Info
    info = {
        "year": 2025,
        "version": "1.0",
        "description": "Generated by Annotation Assistant (Qwen-VL)",
        "date_created": time.strftime("%Y-%m-%d")
    }
    
    # 2. Images
    images = [{
        "id": 1,
        "width": width,
        "height": height,
        "file_name": filename,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    }]
    
    # 3. Categories & Annotations
    categories = []
    category_map = {}
    annotations = []
    cat_id_counter = 1
    
    for i, det in enumerate(detections):
        label = det.get('label', 'object')
        
        # Manage Categories
        if label not in category_map:
            category_map[label] = cat_id_counter
            categories.append({
                "id": cat_id_counter,
                "name": label,
                "supercategory": "object"
            })
            cat_id_counter += 1
            
        x1, y1, x2, y2 = det['box']
        w = x2 - x1
        h = y2 - y1
        
        ann = {
            "id": i + 1,
            "image_id": 1,
            "category_id": category_map[label],
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "attributes": {
                "reasoning": det.get('reasoning', '')
            }
        }
        annotations.append(ann)
        
    coco_output = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": []
    }
    
    return json.dumps(coco_output, indent=2)
