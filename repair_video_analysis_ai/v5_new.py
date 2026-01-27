"""
Complete LLM Advisory Module for Repair Video Analysis
- Vision analysis processing
- Pre-transcribed audio text support (for demo)
- Device metadata integration
- AI-powered repair recommendations with intelligent prompting
"""

import json
import requests
import base64
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import time


@dataclass
class DeviceMetadata:
    """Device information from database/engineer submission."""
    
    category_type: str                    # Air Conditioner, Washing Machine, etc.
    reported_issue: str                   # Gas Leakage, PCB defective, etc.
    device_purchase_date: Optional[str]   # YYYY-MM-DD format
    brand: Optional[str] = None
    model_number: Optional[str] = None
    warranty_status: Optional[str] = None  # Active, Expired, Unknown
    
    def format(self) -> str:
        """Format metadata for prompt."""
        lines = [
            "DEVICE METADATA (from engineer submission):",
            "-" * 80,
            f"  Device Category: {self.category_type}",
            f"  Reported Issue: {self.reported_issue}",
        ]
        
        if self.device_purchase_date:
            lines.append(f"  Purchase Date: {self.device_purchase_date}")
            
            # Calculate age
            try:
                purchase = datetime.strptime(self.device_purchase_date, "%Y-%m-%d")
                age_days = (datetime.now() - purchase).days
                age_years = age_days / 365.25
                lines.append(f"  Device Age: {age_years:.1f} years ({age_days} days)")
            except:
                pass
        
        if self.brand:
            lines.append(f"  Brand: {self.brand}")
        if self.model_number:
            lines.append(f"  Model: {self.model_number}")
        if self.warranty_status:
            lines.append(f"  Warranty: {self.warranty_status}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DeviceMetadata':
        """Create from dictionary (useful for JSON/DB loading)."""
        return cls(
            category_type=data.get('category_type', 'Unknown'),
            reported_issue=data.get('reported_issue', 'Not specified'),
            device_purchase_date=data.get('device_purchase_date'),
            brand=data.get('brand'),
            model_number=data.get('model_number'),
            warranty_status=data.get('warranty_status')
        )
    
    @classmethod
    def from_database(cls, video_id: str, db_connection=None) -> 'DeviceMetadata':
        """
        Fetch metadata from database (placeholder for future implementation).
        
        Args:
            video_id: Video/repair job ID
            db_connection: Database connection object
        
        Returns:
            DeviceMetadata instance
        """
        # TODO: Implement database query when access is available
        # Example:
        # cursor = db_connection.cursor()
        # cursor.execute("SELECT * FROM repair_jobs WHERE video_id = ?", (video_id,))
        # row = cursor.fetchone()
        # return cls.from_dict(row)
        
        raise NotImplementedError("Database access not yet implemented")


@dataclass
class OutputSchema:
    """Define the expected JSON output structure."""
    
    @dataclass
    class DeviceIdentification:
        device_type: str  # AC Unit / Washing Machine / Refrigerator / etc.
        confidence: str   # high / medium / low
        evidence: str
        matches_reported_category: bool  # Does it match metadata?
    
    @dataclass
    class IssueSeverity:
        level: str        # Critical / High / Medium / Low / Informational
        urgency: str      # Immediate / Within 24h / Within Week / Routine
        evidence: str
    
    @dataclass
    class RootCauseAnalysis:
        primary_cause: str
        contributing_factors: List[str]
        aligns_with_reported_issue: bool  # Does analysis match reported issue?
        evidence: str
    
    @dataclass
    class RepairActionPlan:
        immediate_actions: List[str]      # Only critical/urgent actions
        parts_replacement: List[str]      # Only if damage detected
        required_tools: List[str]
        estimated_repair_time: str        # e.g., "30-45 minutes"
        evidence: str
    
    @dataclass
    class ApprovalRecommendation:
        decision: str     # APPROVE / REJECT / NEEDS_REVIEW
        justification: str
        review_conditions: List[str]      # If NEEDS_REVIEW
        cost_impact: str                  # Estimated: Low / Medium / High / Unknown
    
    device_identification: DeviceIdentification
    issue_severity: IssueSeverity
    root_cause_analysis: RootCauseAnalysis
    repair_action_plan: RepairActionPlan
    approval_recommendation: ApprovalRecommendation
    
    @classmethod
    def get_schema_json(cls) -> str:
        """Generate JSON schema representation."""
        return json.dumps({
            "device_identification": {
                "device_type": "string (AC Unit / Washing Machine / Refrigerator / etc.)",
                "confidence": "string (high / medium / low)",
                "evidence": "string (cite specific detections from vision/audio)",
                "matches_reported_category": "boolean (does analysis match engineer's reported category?)"
            },
            "issue_severity": {
                "level": "string (Critical / High / Medium / Low / Informational)",
                "urgency": "string (Immediate / Within 24h / Within Week / Routine)",
                "evidence": "string (cite damage classifications and audio mentions)"
            },
            "root_cause_analysis": {
                "primary_cause": "string (your expert diagnosis based on evidence)",
                "contributing_factors": ["string (list all relevant factors)"],
                "aligns_with_reported_issue": "boolean (does your analysis align with reported issue?)",
                "evidence": "string (cite vision, audio, and metadata that led to this conclusion)"
            },
            "repair_action_plan": {
                "immediate_actions": ["string (ONLY include truly urgent actions)"],
                "parts_replacement": ["string (ONLY if damage is clearly detected)"],
                "required_tools": ["string (specific tools needed)"],
                "estimated_repair_time": "string (realistic time estimate, e.g., '30-45 minutes')",
                "evidence": "string (explain why these actions are necessary)"
            },
            "approval_recommendation": {
                "decision": "string (APPROVE / REJECT / NEEDS_REVIEW)",
                "justification": "string (clear reasoning based on all evidence)",
                "review_conditions": ["string (if NEEDS_REVIEW, what needs clarification?)"],
                "cost_impact": "string (Estimated: Low / Medium / High / Unknown)"
            }
        }, indent=2)


@dataclass
class SystemRules:
    """System instructions for the LLM."""
    
    role: str = "You are an expert HVAC and appliance repair advisor analyzing repair videos and technician reports."
    
    rules: List[str] = None
    
    def __post_init__(self):
        if self.rules is None:
            self.rules = [
                "Base all conclusions on PROVIDED evidence (vision analysis, audio transcript, device metadata).",
                "You MUST analyze causes and provide expert diagnosis - this is your primary job.",
                "Cross-validate findings across all three sources (vision + audio + metadata).",
                "If reported issue doesn't match your analysis, flag this explicitly and explain why.",
                "Be specific and actionable - avoid vague recommendations like 'check the system'.",
                "Only recommend part replacements if damage is clearly visible in vision analysis.",
                "Distinguish between urgent actions and routine maintenance.",
                "If evidence is contradictory or insufficient for a conclusion, state this explicitly.",
                "Consider device age when assessing wear-and-tear vs manufacturing defects.",
                "Output must be valid JSON only. No markdown, no code blocks, no prose.",
            ]
    
    closing: str = "Your analysis will impact approval decisions and cost estimates. Be thorough, evidence-based, and practical."
    
    def format(self) -> str:
        """Format system rules as a clean string."""
        lines = [
            self.role,
            "",
            "ANALYSIS GUIDELINES:"
        ]
        for rule in self.rules:
            lines.append(f"• {rule}")
        lines.append("")
        lines.append(self.closing)
        return "\n".join(lines)


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
                                "Include all technical details, component names, and observations. "
                                "Audio may be in Hindi, English, or mixed. "
                                "Preserve technical terms accurately."
                    },
                    {"inline_data": {"mime_type": mime_type, "data": audio_data}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
        }
        
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


class PromptBuilder:
    """Build structured prompts for LLM."""
    
    def __init__(self, system_rules: SystemRules, output_schema: OutputSchema):
        self.system_rules = system_rules
        self.output_schema = output_schema
    
    def build(self, 
              vision_summary: str, 
              device_metadata: Optional[DeviceMetadata] = None,
              audio_transcript: Optional[str] = None) -> str:
        """Construct the complete prompt."""
        sections = [
            self._format_system_rules(),
            "",
            self._format_separator("INPUT DATA"),
            ""
        ]
        
        # Add metadata first (sets context)
        if device_metadata:
            sections.extend([
                device_metadata.format(),
                "",
                "=" * 80,
                ""
            ])
        
        # Add vision analysis
        sections.extend([
            vision_summary,
            ""
        ])
        
        # Add audio transcript
        if audio_transcript:
            sections.extend([
                self._format_separator("TECHNICIAN'S AUDIO TRANSCRIPT", char="-"),
                audio_transcript,
                "",
                "=" * 80,
                ""
            ])
        
        # Add schema and instructions
        sections.extend([
            self._format_separator("REQUIRED JSON OUTPUT SCHEMA"),
            "",
            self.output_schema.get_schema_json(),
            "",
            self._format_separator("IMPORTANT REMINDERS", char="-"),
            self._format_reminders(),
        ])
        
        return "\n".join(sections)
    
    def _format_system_rules(self) -> str:
        """Format system rules section."""
        return self.system_rules.format()
    
    def _format_separator(self, title: str, char: str = "=") -> str:
        """Create a formatted section separator."""
        line = char * 80
        return f"{line}\n{title}\n{line}"
    
    def _format_reminders(self) -> str:
        """Format critical reminders."""
        reminders = [
            " Cross-reference all three data sources (metadata, vision, audio)",
            " Flag any discrepancies between reported issue and your analysis",
            " Be specific - avoid generic recommendations",
            " Only suggest part replacements if clearly warranted by evidence",
            " Output ONLY valid JSON (no markdown, no code blocks, no extra text)",
        ]
        return "\n".join(reminders) + "\n" + "=" * 80


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
        
        # Initialize schema and prompt builder
        self.system_rules = SystemRules()
        self.output_schema = OutputSchema
        self.prompt_builder = PromptBuilder(self.system_rules, self.output_schema)
        
        print(f"✓ Using Gemini: {model} ({self.model})")
    
    def _prepare_vision_summary(self, vision_results: Dict) -> str:
        """Convert vision JSON to readable summary."""
        
        yolo_detections = vision_results.get('yolo_detections', {})
        damage_classifications = vision_results.get('damage_classifications', {})
        
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
            "VISION ANALYSIS REPORT",
            "-" * 80,
            "",
            "Analysis Models:",
            f"  • Component Detection: {yolo_model}",
            f"  • Damage Classification: {damage_model}",
            "",
            "-" * 80,
            "Detected Components:",
        ]
        
        if part_counts:
            for part, count in sorted(part_counts.items(), key=lambda x: x[1], reverse=True):
                avg_conf = sum(part_confidences[part]) / len(part_confidences[part])
                device = self.PART_TO_DEVICE.get(part, 'Unknown Device')
                
                lines.append(f"\n  • {part.upper()}")
                lines.append(f"    Detections: {count} frames")
                lines.append(f"    Confidence: {avg_conf:.1%}")
                lines.append(f"    Likely Device: {device}")
        else:
            lines.append("\n  [No components detected]")
        
        lines.extend([
            "",
            "-" * 80,
            "Damage Assessment:",
        ])
        
        total = sum(damage_summary.values())
        if total > 0:
            for dtype in ['corrosion', 'leakage', 'normal']:
                if dtype in damage_summary:
                    count = damage_summary[dtype]
                    pct = (count / total) * 100
                    icon = "⚠️" if dtype != 'normal' else "✓"
                    lines.append(f"  {icon} {dtype.upper()}: {count} frames ({pct:.1f}%)")
        
        if critical_frames:
            lines.extend([
                "",
                "-" * 80,
                "High-Priority Damage Detections:",
            ])
            critical_frames.sort(key=lambda x: x['confidence'], reverse=True)
            for i, item in enumerate(critical_frames[:5], 1):
                lines.append(f"  {i}. {item['frame']}: {item['damage'].upper()} (confidence: {item['confidence']:.1%})")
        
        return "\n".join(lines)
    
    def get_recommendations(self, 
                          vision_results: Dict,
                          device_metadata: Optional[DeviceMetadata] = None,
                          audio_transcript: Optional[str] = None) -> Dict:
        """Generate repair recommendations."""
        
        print("\n" + "=" * 80)
        print("GENERATING REPAIR RECOMMENDATIONS")
        print("=" * 80)
        
        vision_summary = self._prepare_vision_summary(vision_results)
        prompt = self.prompt_builder.build(vision_summary, device_metadata, audio_transcript)
        
        print(f"\n[AI] Calling Gemini API...")
        print(f"[AI] Vision data: {len(vision_summary)} chars")
        if device_metadata:
            print(f"[AI] Device metadata: {device_metadata.category_type} - {device_metadata.reported_issue}")
        if audio_transcript:
            print(f"[AI] Audio transcript: {len(audio_transcript)} chars")
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,  # Slightly higher for reasoning
                "maxOutputTokens": 4096
            }
        }
        
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    recommendations = result['candidates'][0]['content']['parts'][0]['text']
                    
                    finish_reason = result['candidates'][0].get('finishReason', '')
                    if finish_reason == 'MAX_TOKENS':
                        print("⚠️  WARNING: Response truncated due to max tokens limit")
                    
                    print(f"✓ Recommendations: {len(recommendations)} chars")
                    print(f"✓ Finish reason: {finish_reason}")
                    
                    return {
                        'status': 'success',
                        'recommendations': recommendations,
                        'vision_summary': vision_summary,
                        'device_metadata': device_metadata.format() if device_metadata else None,
                        'audio_transcript': audio_transcript,
                        'timestamp': datetime.now().isoformat(),
                        'finish_reason': finish_reason
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


def load_metadata_from_file(metadata_file: str) -> DeviceMetadata:
    """Load device metadata from JSON file (temporary solution)."""
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    return DeviceMetadata.from_dict(data)


def process_vision_json(json_file: str, 
                       api_key: str,
                       metadata: Optional[DeviceMetadata] = None,
                       metadata_file: Optional[str] = None,
                       audio_file: Optional[str] = None,
                       pre_transcribed_text: Optional[str] = None,
                       model: str = 'flash-2.5') -> Dict:
    """
    Main function to process vision JSON with optional audio and metadata.
    
    Args:
        json_file: Path to vision analysis JSON
        api_key: Google API key
        metadata: DeviceMetadata object (optional)
        metadata_file: Path to metadata JSON file (optional)
        audio_file: Path to audio file (optional, .mp3 or .wav)
        pre_transcribed_text: Pre-transcribed text (optional)
        model: Gemini model ('flash-2.5', 'flash-2.0', 'pro')
    """
    
    print("\n" + "=" * 80)
    print("REPAIR VIDEO ANALYSIS")
    print("=" * 80)
    
    # Load vision JSON
    print(f"\n[STEP 1] Loading vision analysis...")
    with open(json_file, 'r') as f:
        vision_results = json.load(f)
    
    video_id = vision_results.get('video_id', Path(json_file).stem.replace('_vision', ''))
    print(f"✓ Video ID: {video_id}")
    
    # Load metadata
    if metadata_file and Path(metadata_file).exists():
        print(f"\n[STEP 2] Loading device metadata from file...")
        metadata = load_metadata_from_file(metadata_file)
        print(f"✓ Device: {metadata.category_type}")
        print(f"✓ Reported Issue: {metadata.reported_issue}")
    elif metadata:
        print(f"\n[STEP 2] Using provided metadata...")
        print(f"✓ Device: {metadata.category_type}")
        print(f"✓ Reported Issue: {metadata.reported_issue}")
    else:
        print(f"\n[STEP 2] No metadata provided (will analyze without context)")
    
    # Handle audio transcript
    audio_transcript = None
    
    if pre_transcribed_text:
        print(f"\n[STEP 3] Using pre-transcribed text...")
        audio_transcript = pre_transcribed_text
        print(f"✓ Transcript loaded: {len(audio_transcript)} chars")
    
    elif audio_file and Path(audio_file).exists():
        print(f"\n[STEP 3] Processing audio file...")
        try:
            transcriber = AudioTranscriber(api_key)
            audio_transcript = transcriber.transcribe(audio_file)
            print(f"✓ Transcript complete: {len(audio_transcript)} chars")
            time.sleep(3)
        except Exception as e:
            print(f" Audio transcription failed: {e}")
            print("  Continuing without audio...")
    else:
        print(f"\n[STEP 3] No audio provided")
    
    # Generate recommendations
    print(f"\n[STEP 4] Generating recommendations...")
    advisor = RepairAdvisor(api_key, model)
    recommendations = advisor.get_recommendations(vision_results, metadata, audio_transcript)
    
    if recommendations['status'] != 'success':
        return recommendations
    
    # Save report
    print(f"\n[STEP 5] Saving report...")
    report_path = advisor.save_report(video_id, recommendations)
    
    # Display results
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    if metadata:
        print(f"\nDEVICE INFO:")
        print("=" * 80)
        print(metadata.format())
        print("=" * 80)
    
    if audio_transcript:
        print(f"\nTECHNICIAN TRANSCRIPT:")
        print("=" * 80)
        print(audio_transcript)
        print("=" * 80)
    
    print(f"\nREPAIR RECOMMENDATIONS:")
    print("=" * 80)
    print(recommendations.get('recommendations', ''))
    print("=" * 80)
    
    if recommendations.get('finish_reason') == 'MAX_TOKENS':
        print("\n⚠️  WARNING: Response may have been truncated. Consider using 'pro' model.")
    
    # Save text report
    txt_report = report_path.replace('.json', '.txt')
    with open(txt_report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPAIR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if metadata:
            f.write(metadata.format() + "\n\n")
            f.write("=" * 80 + "\n\n")
        
        if audio_transcript:
            f.write("TECHNICIAN TRANSCRIPT:\n")
            f.write("-" * 80 + "\n")
            f.write(audio_transcript + "\n\n")
            f.write("=" * 80 + "\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(recommendations.get('recommendations', ''))
    
    print(f"\nJSON Report: {report_path}")
    print(f"Text Report: {txt_report}")
    
    return {
        'status': 'success',
        'video_id': video_id,
        'device_metadata': metadata,
        'audio_transcript': audio_transcript,
        'recommendations': recommendations,
        'report_path': report_path
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    GOOGLE_API_KEY = "AIzaSyAQYPW2MoPqVtuHAno4CP9Emfy22bdZEww"
    
    # ========================================================================
    # OPTION 1: Hardcoded metadata (for testing)
    # ========================================================================
    metadata = DeviceMetadata(
        category_type="Air Conditioner-Window AC 1.5 Tons",
        reported_issue="No Cooling",
        device_purchase_date="2023-07-17",
        brand="LG",
        model_number="W3nq18Tnns",
        warranty_status="Expired"
    )
    
    result = process_vision_json(
        json_file=r"C:\Users\Lenovo\Desktop\video-analysis\data\processed\vision\SR_20260123_154910_d96affd0_vision.json",
        api_key=GOOGLE_API_KEY,
        metadata=metadata,  # Pass hardcoded metadata
        model='flash-2.5'
    )
    
    # ========================================================================
    # OPTION 2: Load metadata from JSON file
    # ========================================================================
    # Create example metadata file first:
    # {
    #   "category_type": "Washing Machine",
    #   "reported_issue": "PCB Defective",
    #   "device_purchase_date": "2019-06-20",
    #   "brand": "LG",
    #   "model_number": "T7581NEDL1",
    #   "warranty_status": "Expired"
    # }
    
    # result = process_vision_json(
    #     json_file="path/to/vision.json",
    #     api_key=GOOGLE_API_KEY,
    #     metadata_file="path/to/metadata.json",  # Load from file
    #     model='flash-2.5'
    # )
    
    # ========================================================================
    # OPTION 3: Future - Load from database
    # ========================================================================
    # When database access is available:
    # metadata = DeviceMetadata.from_database(video_id, db_connection)
    # result = process_vision_json(..., metadata=metadata, ...)
    
    if result['status'] == 'success':
        print("\n SUCCESS!")
    else:
        print(f"\nFAILED: {result.get('error')}")