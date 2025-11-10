import os
import json
import dspy
from dotenv import load_dotenv
import pickle
import re
from datetime import datetime

# LOAD ENV VARIABLES
load_dotenv()

# MODEL SETUP
model = dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_tokens=3000
)
dspy.configure(lm=model)

# REMOVE MARKDOWN FUNCTION
def remove_markdown(text):
    text = re.sub(r'(\*\*|\*|__|`)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\-\+\*]\s+', '', text, flags=re.MULTILINE)
    return text.strip()

# SIGNATURE
class PersonaPost(dspy.Signature):
    """Generate a LinkedIn post in the persona's unique style."""
    topic = dspy.InputField(desc="What the post should talk about")
    post_type = dspy.InputField(desc="Type of post: advice, lesson, framework, story, reflection")
    content_points = dspy.InputField(desc="2-3 key points to include")
    post = dspy.OutputField(desc="Generated LinkedIn post")

# LOAD DATASET
with open("data/persona_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def extract_content_points(post_text, topic, post_type):
    lines = post_text.split('\n')
    key_sentences = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 20:
            key_sentences.append(line)
            if len(key_sentences) >= 2:
                break
    if key_sentences:
        return " | ".join([s[:80] for s in key_sentences])
    else:
        return f"{topic} insights | {post_type} perspective | key learnings"

trainset = []
for item in data:
    topic = item.get("topic", "general")
    post_type = item.get("post_type", "advice")
    post_text = remove_markdown(item.get("post", "").strip())
    if not post_text:
        continue
    content_points = extract_content_points(post_text, topic, post_type)
    trainset.append(
        dspy.Example(
            topic=topic,
            post_type=post_type,
            content_points=content_points,
            post=post_text
        ).with_inputs("topic", "post_type", "content_points")
    )

print(f"Loaded {len(trainset)} training examples")

# BASE OPTIMIZER 
from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(
    metric=None,
    max_bootstrapped_demos=5,
    max_labeled_demos=5
)

student = dspy.Predict(PersonaPost)
base_predictor = teleprompter.compile(student, trainset=trainset)

# REFINEMENT CHAIN
# This is the chain step: apply additional instructions for style and LinkedIn optimization
class RefinementChain(dspy.Signature):
    """Refine persona post for style, tone, and clarity."""
    raw_post = dspy.InputField(desc="Base generated post from persona")
    refined_post = dspy.OutputField(desc="Refined LinkedIn-ready post")

refiner = dspy.Predict(RefinementChain)

def chain_predict(topic, post_type, content_points):
    # Base persona generation
    base_result = base_predictor(
        topic=topic,
        post_type=post_type,
        content_points=content_points
    )
    # persona name dynamically
    persona_name = "Ankur Warikoo"  

    # Refinement instructions 
    refinement_prompt = (
    f"Refine the following LinkedIn post in {persona_name}-style:\n\n{base_result.post}\n\n"
    "Instructions:\n"
    "- Keep tone inspirational and conversational\n"
    "- Make post concise and engaging\n"
    "- Include actionable points if possible\n"
    "- Ensure readability for LinkedIn audience\n"
    )
    refined_result = refiner(raw_post=refinement_prompt)
    return refined_result.refined_post

# ================== SAVE MODEL & MEMORY ==================
with open("persona_optimized_chain.pkl", "wb") as f:
    pickle.dump((base_predictor, refiner), f)

# Use current date dynamically
current_date = datetime.now().strftime("%Y-%m-%d")

persona_memory = {
    "persona_name": "Trained Persona",
    "trained_on_topics": list(set([item.topic for item in trainset])),
    "training_date": current_date,
    "post_count": len(trainset)
}

with open("persona_memory_chain.json", "w") as f:
    json.dump(persona_memory, f, indent=2)

print("Persona chain training completed!")
print("Saved: persona_optimized_chain.pkl")
print("Saved: persona_memory_chain.json")
