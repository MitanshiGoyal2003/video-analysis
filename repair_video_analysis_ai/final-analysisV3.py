"""
Complete LLM Advisory Module for Repair Video Analysis
- Vision analysis processing
- Audio transcription (optional)
- AI-powered repair recommendations
"""

import json
import requests
import base64
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import time


class AudioTranscriber:
    """Transcribes audio using Google Gemini."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gemini-2.0-flash-exp"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file (supports Hindi + English)."""
        print(f"\n[AUDIO] Transcribing {Path(audio_path).name}...")
        
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        mime_type = 'audio/mp3' if audio_path.endswith('.mp3') else 'audio/wav'
        
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Transcribe this repair technician's audio. "
                                "Include all technical details. Audio may be Hindi/English/mixed."
                    },
                    {"inline_data": {"mime_type": mime_type, "data": audio_data}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
        }
        
        # Retry logic for rate limits
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    transcript = response.json()['candidates'][0]['content']['parts'][0]['text']
                    print(f"✓ Transcription complete: {len(transcript)} chars")
                    return transcript
                
                elif response.status_code in [429, 503]:
                    wait = (2 ** attempt) * 3
                    print(f"  Rate limit hit. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    response.raise_for_status()
            
            except Exception as e:
                if attempt < 2:
                    print(f"  Error: {e}. Retrying...")
                    time.sleep(5)
                else:
                    raise
        
        raise RuntimeError("Transcription failed after retries")


class RepairAdvisor:
    """Generates repair recommendations using Google Gemini."""
    
    PART_TO_DEVICE = {
        'coil': 'AC Unit (Evaporator/Condenser)',
        'motor': 'AC Unit or Washing Machine',
        'capacitor': 'AC Unit or Washing Machine',
        'compressor': 'AC Unit',
        'fan': 'AC Unit or Washing Machine',
    }
    
    def __init__(self, api_key: str, model: str = 'flash-2.5'):
        self.api_key = api_key
        models = {
            'flash-2.5': 'gemini-2.5-flash',
            'flash-2.0': 'gemini-2.0-flash-exp',
            'pro': 'gemini-2.5-pro',
        }
        self.model = models.get(model, 'gemini-2.5-flash')
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        print(f"✓ Using Gemini: {model} ({self.model})")
    
    def _prepare_vision_summary(self, vision_results: Dict) -> str:
        """Convert vision JSON to readable summary."""
        
        yolo_detections = vision_results.get('yolo_detections', {})
        damage_classifications = vision_results.get('damage_classifications', {})
        
        # Get model info from JSON
        yolo_model = vision_results.get('yolo_model', 'YOLOv8')
        damage_model = vision_results.get('damage_model', 'ResNet')
        
        # Count parts
        part_counts = {}
        part_confidences = {}
        
        for frame_path, detections in yolo_detections.items():
            for det in detections:
                part = det['class_name']
                conf = det['confidence']
                
                if part not in part_counts:
                    part_counts[part] = 0
                    part_confidences[part] = []
                
                part_counts[part] += 1
                part_confidences[part].append(conf)
        
        # Count damage
        damage_summary = {}
        critical_frames = []
        
        for frame_path, classifications in damage_classifications.items():
            if classifications:
                damage_type = classifications[0]['class']
                confidence = classifications[0]['confidence']
                
                damage_summary[damage_type] = damage_summary.get(damage_type, 0) + 1
                
                if confidence >= 0.65 and damage_type != 'normal':
                    critical_frames.append({
                        'frame': Path(frame_path).name,
                        'damage': damage_type,
                        'confidence': confidence
                    })
        
        # Build summary
        lines = [
            "="*80,
            "VISION ANALYSIS REPORT",
            "="*80,
            "",
            "ANALYSIS MODELS:",
            f"  • Component Detection: {yolo_model}",
            f"  • Damage Classification: {damage_model}",
            "",
            "="*80,
            "DETECTED COMPONENTS:",
            "-"*80
        ]
        
        if part_counts:
            for part, count in sorted(part_counts.items(), key=lambda x: x[1], reverse=True):
                avg_conf = sum(part_confidences[part]) / len(part_confidences[part])
                device = self.PART_TO_DEVICE.get(part, 'Unknown Device')
                
                lines.append(f"\n  Component: {part.upper()}")
                lines.append(f"    Frames: {count}")
                lines.append(f"    Avg Confidence: {avg_conf:.1%}")
                lines.append(f"    Device Type: {device}")
        else:
            lines.append("\n  [No components detected]")
        
        lines.extend([
            "",
            "="*80,
            "DAMAGE ASSESSMENT:",
            "-"*80
        ])
        
        total = sum(damage_summary.values())
        if total > 0:
            for dtype in ['corrosion', 'leakage', 'normal']:
                if dtype in damage_summary:
                    count = damage_summary[dtype]
                    pct = (count / total) * 100
                    icon = "⚠️" if dtype != 'normal' else "✓"
                    lines.append(f"\n  {icon} {dtype.upper()}: {count} frames ({pct:.1f}%)")
        
        if critical_frames:
            lines.extend([
                "",
                "="*80,
                "HIGH-PRIORITY DAMAGE FRAMES:",
                "-"*80
            ])
            critical_frames.sort(key=lambda x: x['confidence'], reverse=True)
            for i, item in enumerate(critical_frames[:5], 1):
                lines.append(f"  {i}. {item['frame']}: {item['damage'].upper()} ({item['confidence']:.1%})")
        
        lines.append("\n" + "="*80)
        return "\n".join(lines)
    
    def _build_prompt(self, vision_summary: str, audio_transcript: Optional[str] = None) -> str:
        """Build prompt for Gemini."""
        
        parts = [
            "You are an expert HVAC and appliance repair advisor.",
            "Analyze this field service data and provide repair recommendations.",
            "",
            vision_summary,
            ""
        ]
        
        if audio_transcript:
            parts.extend([
                "="*80,
                "TECHNICIAN'S AUDIO TRANSCRIPT:",
                "-"*80,
                audio_transcript,
                ""
            ])
        
        parts.extend([
            "="*80,
            "PROVIDE COMPREHENSIVE REPAIR ASSESSMENT:",
            "",
            "1. DEVICE IDENTIFICATION",
            "   - Identify the appliance based on detected components",
            "",
            "2. ISSUE SEVERITY",
            "   - Critical/High/Medium/Low",
            "",
            "3. ROOT CAUSE ANALYSIS",
            "   - What caused this issue?",
            "",
            "4. REPAIR ACTION PLAN",
            "   - Step-by-step repair procedures",
            "   - Required parts and tools",
            "",
            "5. APPROVAL RECOMMENDATION",
            "   - YES/NO/REVIEW with justification",
            "",
            "Reference the YOLOv8 detection and ResNet classification results in your analysis.",
            "="*80
        ])
        
        return "\n".join(parts)
    
    def get_recommendations(self, vision_results: Dict, audio_transcript: Optional[str] = None) -> Dict:
        """Generate repair recommendations."""
        
        print("\n" + "="*80)
        print("GENERATING REPAIR RECOMMENDATIONS")
        print("="*80)
        
        vision_summary = self._prepare_vision_summary(vision_results)
        prompt = self._build_prompt(vision_summary, audio_transcript)
        
        print(f"\n[AI] Calling Gemini API...")
        print(f"[AI] Vision data: {len(vision_summary)} chars")
        if audio_transcript:
            print(f"[AI] Audio transcript: {len(audio_transcript)} chars")
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
        }
        
        # Retry logic
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    recommendations = response.json()['candidates'][0]['content']['parts'][0]['text']
                    print(f"✓ Recommendations: {len(recommendations)} chars")
                    
                    return {
                        'status': 'success',
                        'recommendations': recommendations,
                        'vision_summary': vision_summary,
                        'audio_transcript': audio_transcript,
                        'timestamp': datetime.now().isoformat()
                    }
                
                elif response.status_code in [429, 503]:
                    wait = (2 ** attempt) * 3
                    print(f"  API busy. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    response.raise_for_status()
            
            except Exception as e:
                if attempt < 2:
                    print(f"  Error: {e}. Retrying...")
                    time.sleep(5)
                else:
                    return {'status': 'error', 'error': str(e)}
        
        return {'status': 'error', 'error': 'Failed after retries'}
    
    def save_report(self, video_id: str, recommendations: Dict, 
                    output_dir: str = "data/processed/reports") -> str:
        """Save report to JSON file."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"{video_id}_report_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'video_id': video_id,
                'generated_at': datetime.now().isoformat()
            },
            'analysis': recommendations
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Report saved: {report_file}")
        return str(report_file)


def process_vision_json(json_file: str, 
                       api_key: str,
                       audio_file: Optional[str] = None,
                       model: str = 'flash-2.5') -> Dict:
    """
    Main function to process vision JSON with optional audio.
    
    Args:
        json_file: Path to vision analysis JSON
        api_key: Google API key
        audio_file: Path to audio file (optional, .mp3 or .wav)
        model: Gemini model ('flash-2.5', 'flash-2.0', 'pro')
    """
    
    print("\n" + "="*80)
    print("REPAIR VIDEO ANALYSIS")
    print("="*80)
    
    # Step 1: Load vision JSON
    print(f"\n[STEP 1] Loading vision analysis...")
    with open(json_file, 'r') as f:
        vision_results = json.load(f)
    
    video_id = vision_results.get('video_id', Path(json_file).stem.replace('_vision', ''))
    print(f"✓ Video ID: {video_id}")
    
    # Step 2: Transcribe audio (optional)
    audio_transcript = None
    if audio_file and Path(audio_file).exists():
        print(f"\n[STEP 2] Processing audio...")
        try:
            transcriber = AudioTranscriber(api_key)
            audio_transcript = transcriber.transcribe(audio_file)
            
            print(f"\n📝 Transcript Preview:")
            print("-"*80)
            print(audio_transcript[:300] + "..." if len(audio_transcript) > 300 else audio_transcript)
            print("-"*80)
            
            # Wait before next API call
            time.sleep(3)
        
        except Exception as e:
            print(f"⚠️  Audio failed: {e}")
            print("  Continuing with vision only...")
    else:
        print(f"\n[STEP 2] Skipping audio (no file provided)")
    
    # Step 3: Get recommendations
    print(f"\n[STEP 3] Generating recommendations...")
    advisor = RepairAdvisor(api_key, model)
    recommendations = advisor.get_recommendations(vision_results, audio_transcript)
    
    if recommendations['status'] != 'success':
        return recommendations
    
    # Step 4: Save report
    print(f"\n[STEP 4] Saving report...")
    report_path = advisor.save_report(video_id, recommendations)
    
    # Display results
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    if audio_transcript:
        print(f"\n🎤 FULL TRANSCRIPT:")
        print("="*80)
        print(audio_transcript)
        print("="*80)
    
    print(f"\n📋 REPAIR RECOMMENDATIONS:")
    print("="*80)
    print(recommendations['recommendations'])
    print("="*80)
    
    print(f"\n📄 Report: {report_path}")
    
    return {
        'status': 'success',
        'video_id': video_id,
        'audio_transcript': audio_transcript,
        'recommendations': recommendations,
        'report_path': report_path
    }


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    
    GOOGLE_API_KEY = ""
    
    # OPTION 1: Vision + Audio
    result = process_vision_json(
        json_file=r"C:\Users\Lenovo\Desktop\video-analysis\repair_video_analysis_ai\data\processed\vision\SR_20260107_113041_39c80ade_vision.json",
        api_key=GOOGLE_API_KEY,
        audio_file=r"C:\Users\Lenovo\Desktop\video-analysis\repair_video_analysis_ai\data\processed\audio\SR_20260107_112131_39c80ade.wav",
        model='flash-2.5'
    )
    
    # OPTION 2: Vision Only (no audio)
    # result = process_vision_json(
    #     json_file=r"path\to\vision.json",
    #     api_key=GOOGLE_API_KEY,
    #     audio_file=None,  # or just omit this parameter
    #     model='flash-2.5'
    # )
    
    if result['status'] == 'success':
        print("\n✅ SUCCESS!")
    else:
        print(f"\n❌ FAILED: {result.get('error')}")