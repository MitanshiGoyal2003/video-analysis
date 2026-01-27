\# AC/Refrigeration Repair Video Analysis AI



AI-powered system for analyzing AC and refrigeration repair videos to automatically detect spare parts, classify damage types, and generate repair recommendations using computer vision and Google Gemini AI.



\## Features



\- \*\*Video Preprocessing\*\*: Automatic frame extraction and audio separation

\- \*\*YOLO Object Detection\*\*: Detects AC spare parts (coil, motor, capacitor)

\- \*\*Damage Classification\*\*: Classifies damage as normal, corrosion, or leakage

\- \*\*AI Repair Advisor\*\*: Uses Google Gemini to generate detailed repair recommendations

\- \*\*Automated Reports\*\*: JSON reports with confidence scores and repair suggestions



\## Prerequisites



\- Python 3.8 or higher

\- Google Gemini API key (\[Get FREE here](https://aistudio.google.com/apikey))



\## Installation



\### 1. Clone the repository

```bash

git clone https://github.com/yourusername/video-analysis.git

cd video-analysis

```



\### 2. Create virtual environment (recommended)

```bash

\\# Windows

python -m venv venv

venv\\\\Scripts\\\\activate



\\# Linux/Mac

python3 -m venv venv

source venv/bin/activate

```



\### 3. Install dependencies

```bash

pip install -r requirements\\\_full.txt

```



\### 4. Download pre-trained models



\*\*Download these files and place them in your `models/` folder:\*\*



\- \*\*YOLO Model\*\*: \[Download spare\_parts\_v1.pt](https://drive.google.com/file/d/1oABq2\_7qQ549iogM80OdZ9YpYnWfCh9t/view?usp=sharing)

  - Place in: `models/yolo/spare\\\_parts\\\_v1.pt`



\- \*\*Damage Classifier\*\*: \[Download damage\_classifier\_best.pt](https://drive.google.com/file/d/168W9joZg97IyD3hBb5Q4xkZAaiwfzmLx/view?usp=sharing)

  - Place in: `models/damage\\\_classifier/damage\\\_classifier\\\_best.pt`



\### 5. Set up API key



Create a `.env` file in the project root:

```bash

\\# Windows

copy .env.example .env

notepad .env



\\# Linux/Mac

cp .env.example .env

nano .env

```



Add your API key:

```

GOOGLE\\\_API\\\_KEY=your\\\_api\\\_key\\\_here

```



\## 🎯 Usage



\### Quick Start (3 Simple Steps)



\#### Step 1: Add Your Video



Place your AC repair video in the `data/raw/` folder:

```

data/raw/demo\\\_video.mp4

```



\#### Step 2: Preprocess Video



Extract frames and audio:

```bash

python main.py

```



\*\*Output:\*\*

\- Frames: `data/processed/frames/VIDEO\\\_ID/`

\- Audio: `data/processed/audio/VIDEO\\\_ID.wav`



\#### Step 3: Run Vision Analysis



Detect parts and classify damage using pre-trained models:

```bash

python main\\\_vision.py

```



\*\*Output:\*\*

\- Vision analysis JSON: `data/processed/vision/VIDEO\\\_ID\\\_vision.json`



\#### Step 4: Get AI Recommendations



Generate repair recommendations using Google Gemini:

```bash

python final\\\_analysisV2.py

```



\*\*Output:\*\*

\- Repair report: `data/processed/reports/VIDEO\\\_ID\\\_repair\\\_report\\\_TIMESTAMP.json`

\- Console output with full AI recommendations



\## Sample Output

```

REPAIR RECOMMENDATIONS:



1\\. DEVICE IDENTIFICATION

\&nbsp;  - AC Unit with condenser coil

\&nbsp;  - Residential split-type air conditioner



2\\. ISSUE SUMMARY

\&nbsp;  - Primary problem: Refrigerant leakage detected

\&nbsp;  - Severity: HIGH



3\\. ROOT CAUSE ANALYSIS

\&nbsp;  - Corrosion in condenser coil due to environmental exposure

\&nbsp;  - Possible manufacturing defect in coil joints



4\\. RECOMMENDED REPAIR ACTIONS

\&nbsp;  - Replace damaged condenser coil

\&nbsp;  - Perform pressure test after installation

\&nbsp;  - Add anti-corrosion coating



5\\. APPROVAL RECOMMENDATION

\&nbsp;  - Decision: YES - Approve repair

\&nbsp;  - Justification: Critical for system operation

\&nbsp;  - Risk if delayed: Complete system failure, higher costs

```



\##  Project Structure

```

video-analysis/

├── main.py                          # Video preprocessing

├── main\\\_vision.py                   # YOLO + damage detection

├── final\\\_analysisV2.py             # LLM repair advisor

├── src/

│   ├── preprocessing/              # Video processing modules

│   │   ├── video\\\_processor.py

│   │   ├── frame\\\_extractor.py

│   │   └── audio\\\_extractor.py

│   ├── vision/                     # Vision analysis modules

│   │   ├── yolo\\\_detector.py

│   │   ├── damage\\\_classifier.py

│   │   └── vision\\\_pipeline.py

│   └── utils/                      # Helper utilities

│       ├── logger.py

│       └── file\\\_utils.py

├── models/                         # Pre-trained models (download separately)

│   ├── yolo/

│   │   └── spare\\\_parts\\\_v1.pt

│   └── damage\\\_classifier/

│       └── damage\\\_classifier\\\_best.pt

├── data/

│   ├── raw/                        # Input videos

│   └── processed/                  # Output data

│       ├── frames/

│       ├── audio/

│       ├── vision/

│       └── reports/

├── requirements\\\_full.txt

├── .env.example

└── README.md

```



\## 🔧 Training Your Own Models (Optional - Advanced)



If you want to train on your own dataset:



1\. \*\*Prepare dataset\*\* in `dataset/` folder

2\. \*\*Train YOLO model:\*\*

```bash

\&nbsp;  python train\\\_yolo.py

```

3\. \*\*Train damage classifier:\*\*

```bash

\&nbsp;  python train\\\_damage\\\_classifier.py

```



\*\*Note:\*\* Most users don't need this - pre-trained models work out of the box!



\## Troubleshooting



\### "No module named 'ultralytics'"

```bash

pip install ultralytics

```



\### "GOOGLE\_API\_KEY not set"

Make sure you created `.env` file with your API key.



\### "No vision JSON files found"

Run `main\\\_vision.py` before `final\\\_analysisV2.py`.



\### "Model file not found"

Download the pre-trained models (see Installation step 4).



\## API Rate Limits (Google Gemini - FREE Tier)



\- \*\*flash-2.5\*\*: 15 requests/minute ✅ (Recommended)

\- \*\*flash-exp\*\*: 15 requests/minute

\- \*\*pro\*\*: 2 requests/minute

\- \*\*flash-lite\*\*: 15 requests/minute



All models are FREE with generous limits!



\## Contributing



Contributions welcome! Please feel free to submit a Pull Request.



\## 🙏 Acknowledgments



\- YOLOv8 by Ultralytics

\- Google Gemini AI

\- PyTorch and TensorFlow communities



\## 📧 Support



For issues or questions, please open an issue on GitHub.





