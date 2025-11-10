# Persona-Based LinkedIn Writing Agent

This project is a **Persona-Based LinkedIn Post Generator** using DSPy and large language models. It allows you to generate LinkedIn posts in the style of a specific persona by training on example posts.

## Features

- Train a persona model using example LinkedIn posts.
- Generate posts based on topic, type, and key content points.
- Refinement chain to make posts more engaging, readable, and actionable.
- Saves trained model and memory for reuse.

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` packages:
  pip install dspy python-dotenv dill

# A .env file containing your API key:

GOOGLE_API_KEY=your_api_key_here

Folder Structure
.
├── data/                       # Training dataset JSON files
├── generate_post.py             # Script to generate LinkedIn posts
├── train_persona.py             # Script to train the persona model
├── persona_optimized_chain.pkl  # Saved trained model
├── persona_memory_chain.json    # Memory of trained persona
├── .gitignore
└── README.md

Usage

Train the Persona Model

python train_persona.py


Generate a LinkedIn Post

python generate_post.py


Input:

Topic: The topic you want to write about.

Post Type: Advice, lesson, framework, story, reflection.

Content Points: 2-3 key points to include.

Output: A persona-styled LinkedIn post.

Notes

Make sure your .env file is not tracked in GitHub (add it to .gitignore).

This project does not include Streamlit; it's purely CLI-based.
