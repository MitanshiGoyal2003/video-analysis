"""
Complete LLM Advisory Module for Repair Video Analysis
Processes vision analysis JSON files and generates AI-powered repair recommendations.
Uses Google Gemini API (FREE tier with generous limits).
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import os


class RepairAdvisor:
    """
    Generates repair recommendations using Google Gemini API based on:
    - YOLO detected parts (coil, motor, capacitor)
    - Damage classification (normal, corrosion, leakage)
    """
    
    # Mapping of parts to device types
    PART_TO_DEVICE = {
        'coil': 'AC Unit',
        'motor': 'AC Unit or Washing Machine',
        'capacitor': 'AC Unit or Washing Machine'
    }
    
    # Available Gemini models (all FREE with generous limits)
    AVAILABLE_MODELS = {
        'flash-2.5': 'gemini-2.5-flash',         # Latest stable, fast & capable
        'flash': 'gemini-2.0-flash-exp',         # Experimental, FREE
        'pro': 'gemini-2.5-pro',                 # Best quality
        'flash-lite': 'gemini-2.5-flash-lite',   # Ultra-fast, efficient
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'flash-2.5', logger=None):
        """
        Initialize repair advisor with Google Gemini.
        
        Args:
            api_key: Google API key (if None, uses GOOGLE_API_KEY env variable)
            model: Model to use ('flash-2.5', 'flash', 'pro', 'flash-lite')
            logger: Logger instance
        """
        self.logger = logger
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google API key required. Get one FREE at: https://aistudio.google.com/apikey\n"
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model '{model}' not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model = self.AVAILABLE_MODELS[model]
        self.model_name = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        
        if self.logger:
            self.logger.info(f"Repair Advisor initialized with Gemini model: {model}")
        
        print(f"✓ Using Google Gemini model: {model} ({self.model})")
        print(f"✓ FREE tier limits: {'15 req/min' if 'flash' in model else '2 req/min'}")
    
    def _prepare_vision_summary(self, vision_results: Dict) -> str:
        """
        Convert vision analysis results into readable summary.
        Handles large JSON files efficiently by processing frame by frame.
        
        Args:
            vision_results: Output from vision pipeline (can be large)
        
        Returns:
            Formatted string summary (compressed for LLM context)
        """
        yolo_detections = vision_results.get('yolo_detections', {})
        damage_classifications = vision_results.get('damage_classifications', {})
        
        print(f"Processing {len(yolo_detections)} frames of YOLO detections...")
        print(f"Processing {len(damage_classifications)} frames of damage classifications...")
        
        # Count detections by part
        part_counts = {}
        high_confidence_parts = {}
        low_confidence_frames = {}
        
        for frame_path, detections in yolo_detections.items():
            for det in detections:
                part = det['class_name']
                confidence = det['confidence']
                
                part_counts[part] = part_counts.get(part, 0) + 1
                
                if confidence >= 0.6:
                    if part not in high_confidence_parts:
                        high_confidence_parts[part] = []
                    high_confidence_parts[part].append(confidence)
                else:
                    if part not in low_confidence_frames:
                        low_confidence_frames[part] = 0
                    low_confidence_frames[part] += 1
        
        # Count damage types dynamically
        damage_summary = {}
        damage_by_frame = []
        
        for frame_path, classifications in damage_classifications.items():
            if classifications:
                top_class = classifications[0]['class']
                confidence = classifications[0]['confidence']
                
                damage_summary[top_class] = damage_summary.get(top_class, 0) + 1
                
                if confidence >= 0.65 and top_class != 'normal':
                    damage_by_frame.append({
                        'frame': Path(frame_path).name,
                        'damage': top_class,
                        'confidence': confidence
                    })
        
        # Build compressed summary
        summary_lines = []
        summary_lines.append("="*70)
        summary_lines.append("VISION ANALYSIS SUMMARY")
        summary_lines.append("="*70)
        summary_lines.append("")
        
        # Parts detected section
        if part_counts:
            summary_lines.append("DETECTED COMPONENTS:")
            summary_lines.append("-" * 50)
            for part, count in sorted(part_counts.items(), key=lambda x: x[1], reverse=True):
                if part in high_confidence_parts and high_confidence_parts[part]:
                    avg_conf = sum(high_confidence_parts[part]) / len(high_confidence_parts[part])
                    high_conf_count = len(high_confidence_parts[part])
                else:
                    avg_conf = 0.0
                    high_conf_count = 0
                
                device = self.PART_TO_DEVICE.get(part, 'Unknown Device')
                
                summary_lines.append(f"  • {part.upper()}")
                summary_lines.append(f"    - Total detections: {count} frames")
                summary_lines.append(f"    - High confidence (≥0.6): {high_conf_count} frames")
                summary_lines.append(f"    - Average confidence: {avg_conf:.2%}")
                summary_lines.append(f"    - Device type: {device}")
                
                if part in low_confidence_frames:
                    summary_lines.append(f"    - Low confidence frames: {low_confidence_frames[part]}")
                summary_lines.append("")
        else:
            summary_lines.append("No parts detected in this video.")
            summary_lines.append("")
        
        # Damage analysis section
        summary_lines.append("DAMAGE ASSESSMENT:")
        summary_lines.append("-" * 50)
        
        total_frames = sum(damage_summary.values())
        if total_frames > 0:
            severity_order = ['corrosion', 'leakage', 'normal']
            for damage_type in severity_order:
                if damage_type in damage_summary:
                    count = damage_summary[damage_type]
                    percentage = (count / total_frames * 100)
                    severity_icon = "⚠️" if damage_type != 'normal' else "✓"
                    summary_lines.append(
                        f"  {severity_icon} {damage_type.upper()}: "
                        f"{count} frames ({percentage:.1f}%)"
                    )
            
            for damage_type, count in damage_summary.items():
                if damage_type not in severity_order:
                    percentage = (count / total_frames * 100)
                    summary_lines.append(
                        f"  ⚠️ {damage_type.upper()}: "
                        f"{count} frames ({percentage:.1f}%)"
                    )
        else:
            summary_lines.append("  No damage classifications available.")
        
        summary_lines.append("")
        
        # High confidence damage detections
        if damage_by_frame:
            summary_lines.append("CRITICAL DAMAGE FRAMES (Confidence ≥ 65%):")
            summary_lines.append("-" * 50)
            
            damage_by_frame.sort(key=lambda x: x['confidence'], reverse=True)
            for idx, item in enumerate(damage_by_frame[:10], 1):
                summary_lines.append(
                    f"  {idx}. {item['frame']}: "
                    f"{item['damage'].upper()} "
                    f"({item['confidence']:.1%} confidence)"
                )
            
            if len(damage_by_frame) > 10:
                summary_lines.append(f"  ... and {len(damage_by_frame) - 10} more damage detections")
            summary_lines.append("")
        
        summary_lines.append("="*70)
        
        return "\n".join(summary_lines)
    
    def _build_prompt(self, 
                     vision_summary: str,
                     engineer_description: Optional[str] = None) -> str:
        """
        Build the prompt for Gemini API.
        
        Args:
            vision_summary: Summary from vision analysis
            engineer_description: Engineer's verbal description (optional)
        
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("You are an expert HVAC and appliance repair advisor with 20+ years of field experience.")
        prompt_parts.append("Analyze the following technical data from a field service inspection video and provide actionable repair recommendations.")
        prompt_parts.append("")
        prompt_parts.append(vision_summary)
        prompt_parts.append("")
        
        if engineer_description and engineer_description.lower() != 'none':
            prompt_parts.append("TECHNICIAN'S DESCRIPTION:")
            prompt_parts.append("-" * 50)
            prompt_parts.append(engineer_description)
            prompt_parts.append("")
        
        prompt_parts.append("="*70)
        prompt_parts.append("REQUIRED ANALYSIS:")
        prompt_parts.append("="*70)
        prompt_parts.append("")
        prompt_parts.append("Please provide a comprehensive repair assessment with the following sections:")
        prompt_parts.append("")
        prompt_parts.append("1. DEVICE IDENTIFICATION")
        prompt_parts.append("   - What specific device/appliance is this?")
        prompt_parts.append("   - Model type and component details")
        prompt_parts.append("")
        prompt_parts.append("2. ISSUE SUMMARY")
        prompt_parts.append("   - Primary problem detected")
        prompt_parts.append("   - Severity assessment (Critical/High/Medium/Low)")
        prompt_parts.append("")
        prompt_parts.append("3. ROOT CAUSE ANALYSIS")
        prompt_parts.append("   - What likely caused this issue?")
        prompt_parts.append("   - Contributing factors")
        prompt_parts.append("")
        prompt_parts.append("4. RECOMMENDED REPAIR ACTIONS")
        prompt_parts.append("   - Detailed step-by-step repair procedure")
        prompt_parts.append("   - Alternative approaches if applicable")
        prompt_parts.append("")
        prompt_parts.append("5. APPROVAL RECOMMENDATION")
        prompt_parts.append("   - Should this repair be approved? (YES/NO/NEEDS REVIEW)")
        prompt_parts.append("   - Business justification")
        prompt_parts.append("   - Risk of delaying repair")
        prompt_parts.append("")
        prompt_parts.append("Provide clear, professional recommendations suitable for both field technicians and management review.")
        
        return "\n".join(prompt_parts)
    
    def get_recommendations(self,
                           vision_results: Dict,
                           engineer_description: Optional[str] = None) -> Dict:
        """
        Get repair recommendations from Google Gemini API.
        Handles large JSON files efficiently.
        
        Args:
            vision_results: Results from vision pipeline (can be large)
            engineer_description: Engineer's verbal description (optional)
        
        Returns:
            Dictionary with recommendations and metadata
        """
        print("\n" + "="*70)
        print("STARTING LLM REPAIR ANALYSIS")
        print("="*70)
        
        if self.logger:
            self.logger.info("Generating repair recommendations...")
        
        # Step 1: Prepare vision summary
        print("\n[1/3] Preparing vision analysis summary...")
        vision_summary = self._prepare_vision_summary(vision_results)
        print(f"Summary generated ({len(vision_summary)} characters)")
        
        # Step 2: Build prompt
        print("\n[2/3] Building LLM prompt...")
        prompt = self._build_prompt(vision_summary, engineer_description)
        print(f"Prompt ready ({len(prompt)} characters)")
        
        # Step 3: Call Gemini API
        print(f"\n[3/3] Calling Google Gemini API ({self.model_name})...")
        
        try:
            # Gemini API format
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048,
                    "topP": 0.95,
                }
            }
            
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            
            response.raise_for_status()
            
            # Parse Gemini response
            api_response = response.json()
            
            if 'candidates' in api_response and len(api_response['candidates']) > 0:
                recommendations = api_response['candidates'][0]['content']['parts'][0]['text']
            else:
                raise ValueError(f"Unexpected response format: {api_response}")
            
            print(f"✓ Recommendations received ({len(recommendations)} characters)")
            
            if self.logger:
                self.logger.info("Recommendations generated successfully")
            
            return {
                'status': 'success',
                'recommendations': recommendations,
                'vision_summary': vision_summary,
                'engineer_description': engineer_description,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model,
                'prompt_length': len(prompt),
                'response_length': len(recommendations)
            }
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nDetails: {error_detail}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            
            print(f"✗ Error: {error_msg}")
            
            if self.logger:
                self.logger.error(error_msg)
            
            return {
                'status': 'error',
                'error': error_msg,
                'vision_summary': vision_summary,
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_report(self,
                       video_id: str,
                       vision_results: Dict,
                       recommendations: Dict,
                       output_dir: str = "data/processed/reports") -> str:
        """
        Generate a complete repair report and save to file.
        
        Args:
            video_id: Unique video identifier
            vision_results: Vision analysis results
            recommendations: LLM recommendations
            output_dir: Directory to save report
        
        Returns:
            Path to saved report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"{video_id}_repair_report_{timestamp}.json"
        
        # Extract summary statistics
        yolo_detections = vision_results.get('yolo_detections', {})
        damage_classifications = vision_results.get('damage_classifications', {})
        
        report_data = {
            'report_metadata': {
                'video_id': video_id,
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0'
            },
            'video_analysis': {
                'total_frames_analyzed': len(yolo_detections),
                'frames_with_detections': sum(1 for v in yolo_detections.values() if v),
                'total_damage_frames': len(damage_classifications)
            },
            'llm_recommendations': recommendations,
            'raw_vision_data_summary': {
                'yolo_detections_count': len(yolo_detections),
                'damage_classifications_count': len(damage_classifications)
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✓ Report saved: {report_file}")
        
        if self.logger:
            self.logger.info(f"Report saved: {report_file}")
        
        return str(report_file)


def process_vision_json(json_file_path: str,
                       api_key: str,
                       model: str = 'flash-2.5',
                       engineer_description: Optional[str] = None) -> Dict:
    """
    Main function to process a vision analysis JSON file and get LLM recommendations.
    
    Args:
        json_file_path: Path to the vision analysis JSON file
        api_key: Google API key (get FREE at https://aistudio.google.com/apikey)
        model: Model to use ('flash-2.5', 'flash', 'pro', 'flash-lite')
        engineer_description: Optional engineer description
    
    Returns:
        Complete analysis results
    """
    print("\n" + "="*70)
    print("REPAIR VIDEO ANALYSIS - LLM ADVISORY SYSTEM")
    print("="*70)
    print(f"\nProcessing: {json_file_path}")
    
    # Step 1: Load JSON file
    print("\n[STEP 1] Loading vision analysis JSON file...")
    try:
        with open(json_file_path, 'r') as f:
            vision_results = json.load(f)
        
        file_size_mb = Path(json_file_path).stat().st_size / (1024 * 1024)
        print(f"✓ JSON loaded successfully ({file_size_mb:.2f} MB)")
    
    except FileNotFoundError:
        print(f"✗ Error: File not found: {json_file_path}")
        return {'status': 'error', 'error': 'File not found'}
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON format: {str(e)}")
        return {'status': 'error', 'error': 'Invalid JSON'}
    
    # Step 2: Extract video ID
    video_id = vision_results.get('video_id', Path(json_file_path).stem)
    print(f"Video ID: {video_id}")
    
    # Step 3: Initialize advisor
    print(f"\n[STEP 2] Initializing Repair Advisor with Gemini {model}...")
    try:
        advisor = RepairAdvisor(api_key=api_key, model=model)
        print("✓ Advisor initialized")
    except ValueError as e:
        print(f"✗ Error: {str(e)}")
        return {'status': 'error', 'error': str(e)}
    
    # Step 4: Get recommendations
    print("\n[STEP 3] Getting AI recommendations...")
    recommendations = advisor.get_recommendations(
        vision_results=vision_results,
        engineer_description=engineer_description
    )
    
    if recommendations['status'] == 'error':
        print("\n✗ Failed to get recommendations")
        return recommendations
    
    # Step 5: Generate report
    print("\n[STEP 4] Generating final report...")
    report_path = advisor.generate_report(
        video_id=video_id,
        vision_results=vision_results,
        recommendations=recommendations
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nReport saved to: {report_path}")
    
    print("\n" + "="*70)
    print("FULL REPAIR RECOMMENDATIONS:")
    print("="*70)
    rec_text = recommendations['recommendations']
    print(rec_text)  # Print full recommendations, no truncation
    print("="*70)
    
    return {
        'video_id': video_id,
        'status': 'success',
        'recommendations': recommendations,
        'report_path': report_path
    }


# Example usage
if __name__ == "__main__":
    # Configuration
    #JSON_FILE = r"C:\Users\Lenovo\Desktop\video-analysis\repair_video_analysis_ai\data\processed\vision\SR_20260107_113041_39c80ade_vision.json"
    
    vision_dir = Path("repair_video_analysis_ai/data/processed/vision/")
    json_files = sorted(vision_dir.glob("*_vision.json"))
    
    if not json_files:
        print("ERROR: No vision analysis JSON files found!")
        print(f"Please run main_vision.py first")
        print(f"Expected location: {vision_dir}")
        exit(1)
    
    JSON_FILE = str(json_files[-1])  # Use most recent
    print(f"Processing: {JSON_FILE}")
   
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("ERROR: Please set GOOGLE_API_KEY environment variable")
        print("Get your FREE API key at: https://aistudio.google.com/apikey")
        exit(1)
    
    # Available models: 'flash-2.5' (recommended), 'flash' (experimental), 'pro' (best), 'flash-lite' (fastest)
    MODEL = 'flash-2.5'
    
    # Vision-only processing (no audio)
    result = process_vision_json(
        json_file_path=JSON_FILE,
        api_key=GOOGLE_API_KEY,
        model=MODEL,
        engineer_description="None"  # Optional
    )
    
    if result['status'] == 'success':
        print("\n✓ Processing completed successfully!")
        print(f"✓ Report: {result['report_path']}")
    else:
        print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")