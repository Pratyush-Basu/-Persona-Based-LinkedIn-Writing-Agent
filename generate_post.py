import os
import re
import dill as pickle
from dotenv import load_dotenv
import dspy

# LOAD ENV 
load_dotenv()

# MODEL SETUP
dspy.configure(lm=dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_tokens=3000
))

class PersonaPost(dspy.Signature):
    """Generate a LinkedIn post in the persona's unique style."""
    
    topic = dspy.InputField(desc="What the post should talk about")
    post_type = dspy.InputField(desc="Type of post: advice, lesson, framework, story, reflection")
    content_points = dspy.InputField(desc="2-3 key points to include")
    post = dspy.OutputField(desc="Generated LinkedIn post")


class RefinementChain(dspy.Signature):
    """Refine persona post for style, tone, and clarity."""
    
    raw_post = dspy.InputField(desc="Base generated post from persona")
    refined_post = dspy.OutputField(desc="Refined LinkedIn-ready post")


def clean_text(t): 
    return re.sub(r"[*_`'\"]+", "", t.strip())

# LOAD TRAINED MODEL
MODEL_PATH = "persona_optimized_chain.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("persona_optimized_chain.pkl missing. Run training first.")

with open(MODEL_PATH, "rb") as f:
    base_predictor, refiner = pickle.load(f)
print(" Persona model loaded!\n")

# INPUTS
topic = input("Topic: ") or "Mindset and adaptability"
post_type = input("Type (advice/lesson/framework/story): ") or "advice"
content_points = input("Content points: ") or "growth mindset | resilience | self-belief"

# GENERATION
print("\n Generating Persona-based LinkedIn Post...\n")

try:
    base = base_predictor(topic=topic, post_type=post_type, content_points=content_points)
    persona_name = "Ankur Warikoo"
    refined = refiner(raw_post=(
        f"Refine this LinkedIn post in {persona_name}-style:\n\n{base.post}\n\n"
        "Keep it conversational, actionable and engaging."
    ))
    print(" Generated Post:\n")
    print(clean_text(refined.refined_post))
except Exception as e:
    print(f" Error: {e}")
