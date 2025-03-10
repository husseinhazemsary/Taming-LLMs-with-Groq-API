# Taming LLMs with Groq API

## Overview
This project demonstrates how to leverage the Groq API to perform text classification using an LLM (Large Language Model). The script classifies text into predefined categories with confidence scores and compares different prompt engineering strategies.

## Features
- Uses Groq API to perform text classification.
- Implements structured prompt generation for improved analysis.
- Extracts classification results with confidence scores.
- Compares different prompt strategies (basic, structured, and few-shot learning).

## Installation
### Prerequisites
Ensure you have Python installed and set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate    # On Windows
```

### Install Dependencies
Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Setup
Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_actual_api_key_here
```

Make sure the `.env` file is ignored by Git by adding it to `.gitignore`:

```
# Ignore environment files
.env
```

## Usage
Run the script to test text classification and prompt engineering strategies:

```bash
python Lab3.py
```

## File Structure
```
Taming-LLMs-with-Groq-API/
│── .gitignore
│── .env (ignored)
│── Lab3.py (main script)
│── requirements.txt
│── README.md (this file)
```

## Example Output
```
Text: "I love this phone!" -> Category: Positive (Confidence: 0.9)
Reasoning: The sentiment is clearly positive.
...
```

## License
This project is open-source and free to use.

## Author
Hussein Hazem Sabry
