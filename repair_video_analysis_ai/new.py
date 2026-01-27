"""
Complete LLM Advisory Module for Repair Video Analysis
Processes vision analysis JSON files and generates AI-powered repair recommendations.
Includes audio transcript extraction using Gemini.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import os
import subprocess
import ast


class AudioTranscriber:
    """
    Extracts audio transcripts from S3 URLs using Gemini via SSH/Docker.
    """
    
    def __init__(self, ssh_host: str = "ubuntu@iop.qa.onsitego.com", 
                 container: str = "tst7-llm-1",
                 logger=None):
        """
        Initialize audio transcriber.
        
        Args:
            ssh_host: SSH host connection string
            container: Docker container name
            logger: Logger instance
        """
        self.ssh_host = ssh_host
        self.container = container
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Audio Transcriber initialized (SSH: {ssh_host}, Container: {container})")
    
    @staticmethod
    def extract_last_dict(text: str) -> str:
        """
        Extract the last complete dictionary from text output.
        
        Args:
            text: Raw text containing dictionary
            
        Returns:
            String representation of the dictionary
        """
        stack = []
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start:i+1]
        raise ValueError("No complete dict found in output")
    
    def analyse_audio_via_llm(self, s3_url: str) -> str:
        """
        Analyze audio from S3 URL and extract transcript using Gemini.
        
        Args:
            s3_url: S3 URL of the audio file
            
        Returns:
            Transcript text
        """
        if self.logger:
            self.logger.info(f"Analyzing audio from: {s3_url}")
        
        print(f"\n[AUDIO] Connecting to remote server for transcription...")
        print(f"[AUDIO] S3 URL: {s3_url}")
        
        # Build remote Python script
        remote_script = f"""
import json
from gemini.GeminiAnalysisModel import GeminiModel

gm = GeminiModel(
    call_id="local_call",
    model_name="gemini-2.5-flash",
    file_url="{s3_url}",
    model_temperature="0.1",
    insight_prompt="give me the audio transcript",
    is_thinking_model=False,
    thinking_tokens_count=0
)

resp = gm.analyse_audio()
print(resp)
"""
        
        # Build SSH command
        ssh_command = f"""
ssh -T -o LogLevel=ERROR {self.ssh_host} << 'EOF'
docker exec -i {self.container} python << 'PYEOF'
{remote_script}
PYEOF
EOF
"""
        
        try:
            # Execute SSH command
            print(f"[AUDIO] Executing remote transcription...")
            result = subprocess.run(
                ssh_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=180
            )
            
            if result.returncode != 0:
                error_msg = f"SSH command failed: {result.stderr}"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Parse output
            stdout = result.stdout.strip()
            
            # Extract outer Python dict
            outer_str = self.extract_last_dict(stdout)
            
            try:
                outer = json.loads(outer_str)
            except json.JSONDecodeError:
                outer = ast.literal_eval(outer_str)
            
            # Parse inner JSON
            inner = json.loads(outer["data"])
            transcript = inner["transcript"]
            
            print(f"[AUDIO] ✓ Transcript extracted ({len(transcript)} characters)")
            
            if self.logger:
                self.logger.info(f"Audio transcript extracted successfully")
            
            return transcript
            
        except subprocess.TimeoutExpired:
            error_msg = "Audio transcription timed out (180s)"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Audio transcription failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def extract_keywords(self, transcript: str, top_n: int = 10) -> List[str]:
        """
        Extract relevant keywords from transcript.
        Simple keyword extraction based on common repair terms.
        
        Args:
            transcript: Full transcript text
            top_n: Number of top keywords to extract
            
        Returns:
            List of keywords
        """
        # Common repair-related keywords
        repair_terms = [
            'coil', 'motor', 'capacitor', 'compressor', 'fan',
            'corrosion', 'leakage', 'damage', 'broken', 'replace',
            'repair', 'fix', 'AC', 'unit', 'washing machine',
            'refrigerant', 'leak', 'rust', 'worn', 'faulty'
        ]
        
        # Convert to lowercase
        transcript_lower = transcript.lower()
        
        # Find matching keywords
        found_keywords = []
        for term in repair_terms:
            if term in transcript_lower:
                found_keywords.append(term)
        
        return found_keywords[:top_n]


class RepairAdvisor:
    """
    Generates repair recommendations using Claude API based on:
    - YOLO detected parts (coil, motor, capacitor)
    - Damage classification (normal, corrosion, leakage)
    - Audio transcription keywords
    """
    
    # Mapping of parts to device types
    PART_TO_DEVICE = {
        'coil': 'AC Unit',
        'motor': 'AC Unit or Washing Machine',
        'capacitor': 'AC Unit or Washing Machine'
    }
    
    def __init__(self, api_key: Optional[str] = None, logger=None):
        """
        Initialize repair advisor.
        
        Args:
            api_key: Anthropic API key (if None, uses environment variable)
            logger: Logger instance
        """
        self.logger = logger
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.api_url = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        if self.logger:
            self.logger.info("Repair Advisor initialized")
    
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
                    # Track low confidence detections
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
                
                # Dynamic counting for any damage type
                damage_summary[top_class] = damage_summary.get(top_class, 0) + 1
                
                # Collect high confidence damage (lowered threshold to 0.65)
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
            # Sort by severity
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
            
            # Add any other damage types
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
            
            # Sort by confidence descending and show top 10
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
                     audio_keywords: Optional[List[str]] = None,
                     engineer_description: Optional[str] = None) -> str:
        """
        Build the prompt for Claude API.
        
        Args:
            vision_summary: Summary from vision analysis
            audio_keywords: Keywords extracted from audio
            engineer_description: Engineer's verbal description
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("You are an expert HVAC and appliance repair advisor with 20+ years of field experience.")
        prompt_parts.append("Analyze the following technical data from a field service inspection video and provide actionable repair recommendations.")
        prompt_parts.append("")
        prompt_parts.append(vision_summary)
        prompt_parts.append("")
        
        if audio_keywords:
            prompt_parts.append("AUDIO KEYWORDS DETECTED FROM TECHNICIAN:")
            prompt_parts.append("-" * 50)
            prompt_parts.append(", ".join(audio_keywords))
            prompt_parts.append("")
        
        if engineer_description:
            prompt_parts.append("TECHNICIAN'S VERBAL DESCRIPTION:")
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
        prompt_parts.append("5. REQUIRED PARTS & TOOLS")
        prompt_parts.append("   - Complete list of replacement parts")
        prompt_parts.append("   - Specialized tools needed")
        prompt_parts.append("")
        prompt_parts.append("6. TIME & COST ESTIMATE")
        prompt_parts.append("   - Estimated repair duration")
        prompt_parts.append("   - Approximate parts cost range")
        prompt_parts.append("")
        prompt_parts.append("7. SAFETY WARNINGS")
        prompt_parts.append("   - Critical safety precautions")
        prompt_parts.append("   - Required certifications or licenses")
        prompt_parts.append("")
        prompt_parts.append("8. APPROVAL RECOMMENDATION")
        prompt_parts.append("   - Should this repair be approved? (YES/NO/NEEDS REVIEW)")
        prompt_parts.append("   - Business justification")
        prompt_parts.append("   - Risk of delaying repair")
        prompt_parts.append("")
        prompt_parts.append("Provide clear, professional recommendations suitable for both field technicians and management review.")
        
        return "\n".join(prompt_parts)
    
    def get_recommendations(self,
                          vision_results: Dict,
                          audio_keywords: Optional[List[str]] = None,
                          engineer_description: Optional[str] = None) -> Dict:
        """
        Get repair recommendations from Claude API.
        Handles large JSON files efficiently.
        
        Args:
            vision_results: Results from vision pipeline (can be large)
            audio_keywords: Keywords from audio transcription
            engineer_description: Engineer's verbal description
            
        Returns:
            Dictionary with recommendations and metadata
        """
        print("\n" + "="*70)
        print("STARTING LLM REPAIR ANALYSIS")
        print("="*70)
        
        if self.logger:
            self.logger.info("Generating repair recommendations...")
        
        # Step 1: Prepare vision summary (compresses large JSON)
        print("\n[1/3] Preparing vision analysis summary...")
        vision_summary = self._prepare_vision_summary(vision_results)
        print(f"✓ Summary generated ({len(vision_summary)} characters)")
        
        # Step 2: Build prompt
        print("\n[2/3] Building LLM prompt...")
        prompt = self._build_prompt(vision_summary, audio_keywords, engineer_description)
        print(f"✓ Prompt ready ({len(prompt)} characters)")
        
        # Step 3: Call Claude API
        print("\n[3/3] Calling Claude API...")
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,  # Increased for detailed response
                    "temperature": 0.3,  # Lower temperature for more focused output
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=60  # Increased timeout for longer processing
            )
            
            response.raise_for_status()
            
            # Parse response
            api_response = response.json()
            recommendations = api_response['content'][0]['text']
            
            print(f"✓ Recommendations received ({len(recommendations)} characters)")
            
            if self.logger:
                self.logger.info("Recommendations generated successfully")
            
            return {
                'status': 'success',
                'recommendations': recommendations,
                'vision_summary': vision_summary,
                'audio_keywords': audio_keywords,
                'engineer_description': engineer_description,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'claude-sonnet-4-20250514',
                'prompt_length': len(prompt),
                'response_length': len(recommendations)
            }
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
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
                       audio_s3_url: Optional[str] = None,
                       ssh_host: str = "ubuntu@iop.qa.onsitego.com",
                       container: str = "tst7-llm-1",
                       engineer_description: Optional[str] = None) -> Dict:
    """
    Main function to process a vision analysis JSON file and get LLM recommendations.
    Optionally includes audio transcript extraction.
    
    Args:
        json_file_path: Path to the vision analysis JSON file
        api_key: Anthropic API key
        audio_s3_url: Optional S3 URL for audio file
        ssh_host: SSH host for audio transcription
        container: Docker container for audio transcription
        engineer_description: Optional engineer description (overrides audio)
        
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
    print(f"✓ Video ID: {video_id}")
    
    # Step 3: Process audio (if S3 URL provided)
    audio_keywords = None
    audio_transcript = None
    
    if audio_s3_url:
        print("\n[STEP 2] Processing audio transcript...")
        try:
            transcriber = AudioTranscriber(ssh_host=ssh_host, container=container)
            audio_transcript = transcriber.analyse_audio_via_llm(audio_s3_url)
            audio_keywords = transcriber.extract_keywords(audio_transcript)
            
            print(f"✓ Transcript extracted: {len(audio_transcript)} characters")
            print(f"✓ Keywords found: {', '.join(audio_keywords)}")
            
            # Use transcript as engineer description if not provided
            if not engineer_description:
                engineer_description = audio_transcript
                
        except Exception as e:
            print(f"⚠ Warning: Audio transcription failed: {str(e)}")
            print("  Continuing without audio data...")
    else:
        print("\n[STEP 2] No audio URL provided, skipping audio transcription")
    
    # Step 4: Initialize advisor
    step_num = 3 if audio_s3_url else 2
    print(f"\n[STEP {step_num}] Initializing Repair Advisor...")
    try:
        advisor = RepairAdvisor(api_key=api_key)
        print("✓ Advisor initialized")
    except ValueError as e:
        print(f"✗ Error: {str(e)}")
        return {'status': 'error', 'error': str(e)}
    
    # Step 5: Get recommendations
    step_num += 1
    print(f"\n[STEP {step_num}] Getting AI recommendations...")
    recommendations = advisor.get_recommendations(
        vision_results=vision_results,
        audio_keywords=audio_keywords,
        engineer_description=engineer_description
    )
    
    if recommendations['status'] == 'error':
        print("\n✗ Failed to get recommendations")
        return recommendations
    
    # Step 6: Generate report
    step_num += 1
    print(f"\n[STEP {step_num}] Generating final report...")
    report_path = advisor.generate_report(
        video_id=video_id,
        vision_results=vision_results,
        recommendations=recommendations
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nReport saved to: {report_path}")
    
    if audio_transcript:
        print(f"\nAudio Transcript Preview:")
        print("-" * 70)
        print(audio_transcript[:300] + "..." if len(audio_transcript) > 300 else audio_transcript)
    
    print("\nRecommendations Preview:")
    print("-" * 70)
    print(recommendations['recommendations'][:500] + "...")
    
    return {
        'video_id': video_id,
        'status': 'success',
        'audio_transcript': audio_transcript,
        'audio_keywords': audio_keywords,
        'recommendations': recommendations,
        'report_path': report_path
    }


# Example usage
if __name__ == "__main__":
    # Configuration
    JSON_FILE = "data/processed/vision_analysis/SR_20260107_113041_39c80ade_analysis.json"
    API_KEY = "your-anthropic-api-key-here"  # Or use environment variable
    
    # Option 1: With audio S3 URL (recommended)
    AUDIO_S3_URL = "s3://your-bucket/audio/repair_video_audio.mp3"
    
    result = process_vision_json(
        json_file_path=JSON_FILE,
        api_key=API_KEY,
        audio_s3_url=AUDIO_S3_URL,  # Will auto-extract transcript and keywords
        ssh_host="ubuntu@iop.qa.onsitego.com",
        container="tst7-llm-1"
    )
    
    # Option 2: Without audio (vision only)
    # result = process_vision_json(
    #     json_file_path=JSON_FILE,
    #     api_key=API_KEY,
    #     engineer_description="Manual description if needed"
    # )
    
    # Option 3: Manual audio keywords (if you already have them)
    # result = process_vision_json(
    #     json_file_path=JSON_FILE,
    #     api_key=API_KEY,
    #     engineer_description="AC unit coil has severe corrosion"
    # )
    
    if result['status'] == 'success':
        print("\n✓ Processing completed successfully!")
        print(f"\nFull transcript available in result['audio_transcript']")
        print(f"Keywords: {result.get('audio_keywords', [])}")
    else:
        print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")