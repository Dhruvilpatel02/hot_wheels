# methods.py - Enhanced with Better Search and Concurrency

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from datetime import datetime
import requests
import json
import base64
from duckduckgo_search import DDGS
import time
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Configuration with Multiple API Keys and Failover
# ----------------------------

class GeminiAPIManager:
    """Manages multiple Gemini API keys with automatic failover on quota errors"""
    
    def __init__(self):
        # Directly integrated API keys
        self.api_keys = [
            "AIzaSyBI6ZrFncJ7V7ezsyNBdn9N8Q1RFTdDlS4",
            "AIzaSyDLKQEgbVETZSFptZwlFoLbUg_DEFUlnjs",
            "AIzaSyAFQCIi1KnuMuBjCD8VVZ9q5kEX14EuEP8",
            "AIzaSyAQbJYBKpCKZrFvKkirjotdBMbpN6ZNzJg",
            "AIzaSyB31lni3hnZ9f__CC4EK1RDG8KCLS_YJXo"
        ]
        self.current_key_index = 0
        self.failed_keys = set()
        self.key_usage_count = {key: 0 for key in self.api_keys}
        self.last_error_time = {key: 0 for key in self.api_keys}
        
    def get_current_key(self):
        """Get the current active API key"""
        if self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return None
    
    def get_next_available_key(self):
        """Find and switch to the next available API key"""
        current_time = time.time()
        for key in list(self.failed_keys):
            if current_time - self.last_error_time.get(key, 0) > 3600:
                self.failed_keys.remove(key)
                st.info(f"🔄 Retrying previously failed API key after cooldown")
        
        for i in range(len(self.api_keys)):
            if self.api_keys[i] not in self.failed_keys:
                self.current_key_index = i
                st.success(f"✅ Switched to API key #{i+1} of {len(self.api_keys)}")
                return self.api_keys[i]
        
        st.error("❌ All API keys have reached their quota limits")
        return None
    
    def mark_key_failed(self, key, error_msg=""):
        """Mark a key as failed and switch to next one"""
        self.failed_keys.add(key)
        self.last_error_time[key] = time.time()
        key_index = self.api_keys.index(key) + 1 if key in self.api_keys else 0
        
        st.warning(f"⚠️ API key #{key_index} hit quota limit: {error_msg}")
        st.info(f"🔄 Attempting to switch to next available API key...")
        
        next_key = self.get_next_available_key()
        return next_key
    
    def increment_usage(self, key):
        """Track usage count for monitoring"""
        if key in self.key_usage_count:
            self.key_usage_count[key] += 1
    
    def get_status(self):
        """Get current status of all API keys"""
        status = []
        for i, key in enumerate(self.api_keys, 1):
            key_id = f"Key #{i}"
            usage = f"(Used {self.key_usage_count[key]} times)"
            if key in self.failed_keys:
                status.append(f"{key_id}: ❌ Quota exceeded")
            elif i == self.current_key_index + 1:
                status.append(f"{key_id}: ✅ Active {usage}")
            else:
                status.append(f"{key_id}: ⏸️ Available {usage}")
        return status

api_manager = GeminiAPIManager()

# ----------------------------
# BBox, Image, and File Helpers
# ----------------------------

def parse_bbox_json(json_path: str) -> List[Dict]:
    """Parses a JSON file containing bounding box data."""
    detections = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            coords = item.get("box_coordinates", {})
            x1, y1, x2, y2 = coords.get("x1"), coords.get("y1"), coords.get("x2"), coords.get("y2")
            
            if all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
                detections.append({
                    "class_name": item.get("class_name", "unknown"),
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error reading bounding box JSON file '{json_path}': {e}")
    return detections

def draw_bboxes_rgb(rgb_img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draws bounding boxes on an image."""
    out = rgb_img.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        label = f"Car #{i+1}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(bgr, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def crop_and_upscale(rgb_img: np.ndarray, detections: List[Dict], padding: int = 0, upscale_factor: int = 3, do_sharpen: bool = True) -> List[np.ndarray]:
    """Crops, upscales, and optionally sharpens detected regions."""
    H, W = rgb_img.shape[:2]
    crops = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
        x2p, y2p = min(W, x2 + padding), min(H, y2 + padding)
        
        crop = rgb_img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
            
        new_w, new_h = max(1, int(crop.shape[1] * upscale_factor)), max(1, int(crop.shape[0] * upscale_factor))
        up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        if do_sharpen:
            up_bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
            blur = cv2.GaussianBlur(up_bgr, (0, 0), sigmaX=1.2)
            sharp_bgr = cv2.addWeighted(up_bgr, 1.25, blur, -0.25, 0)
            up = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB)
            
        crops.append(up)
    return crops

# ----------------------------
# Enhanced Gemini Vision Integration
# ----------------------------
def encode_image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64."""
    pil_img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode()

def call_gemini_vision(image_array: np.ndarray, prompt: str, temperature: float = 0.2, retry_count: int = 0) -> str:
    """Call Gemini Vision API with automatic API key failover."""
    api_key = api_manager.get_current_key()
    if not api_key:
        return "Error: No API keys available"
    
    # <<< START: MODIFIED SECTION >>>
    # UPDATED to the correct, modern model URL for multimodal input
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    # <<< END: MODIFIED SECTION >>>

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    base64_image = encode_image_to_base64(image_array)
    
    data = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    }
    
    try:
        r = requests.post(url, headers=headers, params=params, json=data, timeout=45)
        
        if r.status_code == 429 or "quota" in r.text.lower():
            next_key = api_manager.mark_key_failed(api_key, f"Status {r.status_code}")
            if next_key and retry_count < len(api_manager.api_keys):
                st.info("🔄 Retrying with new API key...")
                return call_gemini_vision(image_array, prompt, temperature, retry_count + 1)
            return "Error: All API keys exhausted."
        
        r.raise_for_status() # This will raise an exception for 4xx or 5xx errors
        out = r.json()

        if "candidates" in out and out["candidates"]:
            api_manager.increment_usage(api_key)
            return out["candidates"][0]["content"]["parts"][0]["text"]
        
        # Handle cases where the model returns a safety block or empty response
        error_info = out.get("promptFeedback", "No detailed error provided.")
        return f"Gemini API response error: {error_info}"

    except requests.exceptions.RequestException as e:
        if "quota" in str(e).lower():
            next_key = api_manager.mark_key_failed(api_key, str(e))
            if next_key and retry_count < len(api_manager.api_keys):
                return call_gemini_vision(image_array, prompt, temperature, retry_count + 1)
        if retry_count < 2:
            time.sleep(1)
            return call_gemini_vision(image_array, prompt, temperature, retry_count + 1)
        return f"Error calling Gemini API: {e}"

def call_gemini_text(prompt: str, temperature: float = 0.3, retry_count: int = 0) -> str:
    """Call Gemini text generation API with automatic failover."""
    api_key = api_manager.get_current_key()
    if not api_key:
        return "Error: No API keys available"

    # <<< START: MODIFIED SECTION >>>
    # UPDATED to the correct, modern model URL for text-only input
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    # <<< END: MODIFIED SECTION >>>

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}
    
    try:
        r = requests.post(url, headers=headers, params=params, json=data, timeout=45)
        
        if r.status_code == 429 or "quota" in r.text.lower():
            next_key = api_manager.mark_key_failed(api_key, f"Status {r.status_code}")
            if next_key and retry_count < len(api_manager.api_keys):
                return call_gemini_text(prompt, temperature, retry_count + 1)
            return "Error: All API keys exhausted."
            
        r.raise_for_status()
        out = r.json()

        if "candidates" in out and out["candidates"]:
            api_manager.increment_usage(api_key)
            return out["candidates"][0]["content"]["parts"][0]["text"]

        error_info = out.get("promptFeedback", "No detailed error provided.")
        return f"Gemini API response error: {error_info}"

    except requests.exceptions.RequestException as e:
        if "quota" in str(e).lower():
            next_key = api_manager.mark_key_failed(api_key, str(e))
            if next_key and retry_count < len(api_manager.api_keys):
                return call_gemini_text(prompt, temperature, retry_count + 1)
        if retry_count < 2:
            time.sleep(1)
            return call_gemini_text(prompt, temperature, retry_count + 1)
        return f"Error calling Gemini API: {e}"

# ----------------------------
# Enhanced Hot Wheels Search System
# ----------------------------

class EnhancedHotWheelsSearchEngine:
    """Enhanced search engine with comprehensive database coverage and concurrency"""
    def __init__(self):
        self.max_workers = 6
        self.hotwheels_websites = {
            "hotwheels.fandom.com": {"priority": 10}, "hobbydb.com": {"priority": 9},
            "southtexasdiecast.com": {"priority": 8}, "hwcollectorsnews.com": {"priority": 8},
            "orangetrackdiecast.com": {"priority": 8}, "hotwheels.mattel.com": {"priority": 10},
            "lamleygroup.com": {"priority": 7}, "ebay.com": {"priority": 5},
            "worthpoint.com": {"priority": 6}
        }
    
    def search_single_site_concurrent(self, site: str, query: str) -> List[Dict]:
        """Search a single site for concurrent execution"""
        try:
            with DDGS() as ddgs:
                search_query = f'site:{site} Hot Wheels "{query}"'
                return list(ddgs.text(search_query, max_results=3))
        except Exception:
            return [] # Fail silently

    def concurrent_hotwheels_search(self, query: str, max_sites: int = 12) -> List[Dict]:
        """Perform concurrent searches across multiple Hot Wheels databases"""
        all_results = []
        priority_sites = sorted(self.hotwheels_websites.keys(), 
                                key=lambda s: self.hotwheels_websites[s]['priority'], 
                                reverse=True)[:max_sites]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_site = {executor.submit(self.search_single_site_concurrent, site, query): site for site in priority_sites}
            for future in as_completed(future_to_site):
                try:
                    all_results.extend(future.result(timeout=10))
                except Exception:
                    pass # Continue if a site fails or times out
        return all_results
    
    def multi_strategy_enhanced_search(self, base_query: str) -> List[Dict]:
        """Enhanced multi-strategy search"""
        all_results = self.concurrent_hotwheels_search(base_query, max_sites=12)
        
        # Deduplicate results based on URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get('href')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        return unique_results

# ----------------------------
# Enhanced Car Identification Pipeline
# ----------------------------
def ultra_precise_car_identification_enhanced(image_array: np.ndarray) -> Dict:
    """Enhanced car identification with improved prompts"""
    enhanced_prompt = (
        "You are a world-leading Hot Wheels expert. Analyze this die-cast car image with extreme precision. "
        "Provide your best expert analysis based on visible body shape, tampos, wheel design, and color. "
        "Return a structured analysis using these exact fields:\n"
        "**HOT WHEELS CASTING NAME**: <your best identification or 'Unidentified'>\n"
        "**REAL CAR MODEL**: <actual vehicle this represents>\n"
        "**FIRST RELEASE YEAR**: <year or best estimate>\n"
        "**COLOR/FINISH**: <detailed color description>\n"
        "**SERIES/LINE**: <mainline, premium, series name>\n"
        "**IDENTIFICATION CONFIDENCE**: <High/Medium/Low> - <explanation>\n"
        "**SEARCH KEYWORDS**: <optimal terms for database searches>\n"
    )
    analysis = call_gemini_vision(image_array, enhanced_prompt, 0.05)
    return {"enhanced_analysis": analysis, "timestamp": datetime.now().isoformat()}

def synthesize_enhanced_collector_report(identification_data: Dict, search_results: List[Dict], search_terms: str) -> str:
    """Generate enhanced collector report with better synthesis"""
    synthesis_prompt = f"""
Create a definitive Hot Wheels collector identification report.

VISUAL ANALYSIS:
{identification_data.get('enhanced_analysis', '')}

SEARCH TERMS USED: {search_terms}
SEARCH RESULTS (Top 5):
{[f"- {r.get('title', '')}: {r.get('body', '')[:150]}..." for r in search_results[:5]]}

Generate a comprehensive report with these sections:

## 🎯 **Definitive Identification**
- **Hot Wheels Casting Name**: [Most accurate name based on all sources]
- **Real Vehicle Basis**: [Actual car this represents]
- **First Release Year**: [When first introduced]
- **This Variant Year**: [Specific year of this version]
- **Identification Confidence**: [High/Medium/Low with reasoning]

## 📊 **Collector Specifications**
- **Series/Line**: [Mainline, Premium, specific series]
- **Color & Finish**: [Detailed color description]
- **Wheel Type**: [Specific wheel design]

## 💰 **Market Value & Rarity**
- **Rarity Level**: [Common/Uncommon/Rare/Treasure Hunt]
- **Estimated Value**: [Provide a general estimate, e.g., ₹80-250 for common, etc.]
"""
    return call_gemini_text(synthesis_prompt, 0.1)

def extract_confidence_level(analysis_text: str) -> str:
    """Extract confidence level from analysis"""
    match = re.search(r"\*\*IDENTIFICATION CONFIDENCE\*\*:\s*(\w+)", analysis_text, re.IGNORECASE)
    return match.group(1) if match else "Unknown"

def extract_quick_display_info(analysis_text: str) -> Dict:
    """Extract key information for quick display"""
    info = {}
    patterns = {
        "Casting Name": r"\*\*HOT WHEELS CASTING NAME\*\*:\s*(.+)", "Real Car": r"\*\*REAL CAR MODEL\*\*:\s*(.+)",
        "Color/Finish": r"\*\*COLOR/FINISH\*\*:\s*(.+)", "First Released": r"\*\*FIRST RELEASE YEAR\*\*:\s*(.+)",
        "Confidence": r"\*\*IDENTIFICATION CONFIDENCE\*\*:\s*(.+)", "Series": r"\*\*SERIES/LINE\*\*:\s*(.+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, analysis_text, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip().split('\n')[0]
            if " - " in value: value = value.split(" - ")[0]
            info[key] = value
        else:
            info[key] = "N/A"
    return info

# ----------------------------
# Main Concurrent Analysis Pipeline
# ----------------------------
def analyze_single_car(crop_data: Tuple[np.ndarray, int]) -> Dict:
    """Analyzes a single car image, designed for concurrent execution."""
    crop_image, crop_number = crop_data
    search_engine = EnhancedHotWheelsSearchEngine()
    
    identification_data = ultra_precise_car_identification_enhanced(crop_image)
    
    stage1_text = identification_data.get('enhanced_analysis', '')
    match = re.search(r"\*\*SEARCH KEYWORDS\*\*:\s*(.+)", stage1_text)
    primary_search_term = match.group(1).strip() if match else "Hot Wheels die cast car"
    
    search_results = search_engine.multi_strategy_enhanced_search(primary_search_term)
    
    comprehensive_report = synthesize_enhanced_collector_report(
        identification_data, search_results, primary_search_term
    )
    
    return {
        "crop_number": crop_number,
        "identification_data": identification_data,
        "search_terms": primary_search_term,
        "search_results": search_results,
        "comprehensive_report": comprehensive_report,
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_sources": len(search_results),
        "confidence_level": extract_confidence_level(stage1_text),
        "quick_identification": extract_quick_display_info(stage1_text) # Add for DB storage
    }

def process_all_cars_enhanced(api_key: str, crops: List[np.ndarray]) -> List[Dict]:
    """Main enhanced processing function with concurrent analysis"""
    # Note: api_key is passed but not used directly here, as api_manager handles it.
    st.write(f"📊 **Starting Concurrent Analysis for {len(crops)} cars...**")
    
    all_results = []
    crop_data_list = list(enumerate(crops, 1))
    
    max_workers = min(4, len(crops))
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_crop = {executor.submit(analyze_single_car, (crop, num)): num for num, crop in crop_data_list}
        
        for future in as_completed(future_to_crop):
            crop_number = future_to_crop[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                st.error(f"❌ Error analyzing Car #{crop_number}: {e}")
                all_results.append({
                    "crop_number": crop_number, "comprehensive_report": f"Analysis failed: {e}",
                    "confidence_level": "Error", "quick_identification": {}
                })
            
            completed_count += 1
            progress = completed_count / len(crops)
            progress_bar.progress(progress)
            status_text.text(f"🏆 Completed Car {completed_count} of {len(crops)}")

    progress_bar.empty()
    status_text.empty()
    
    all_results.sort(key=lambda x: x['crop_number'])
    return all_results