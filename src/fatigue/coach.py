import logging
import os

import litellm
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

log = logging.getLogger("gcp-api")


class FatigueCoach:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if self.api_key:
            litellm.api_key = self.api_key
            log.info("Groq API Key successfully loaded into LiteLLM.")
        else:
            log.warning("GROQ_API_KEY NOT FOUND in environment variables!")
        # 1. Local Free Embedding Model (Runs on GCP CPU)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # 2. Predefined Knowledge Base
        self.courses = [
            {
                "id": "acute_recovery",
                "topic": "Acute Fatigue Recovery",
                "text": "10-20 min power nap. Eyes closed, no phone. Avoid caffeine 6 hours before main sleep. A short rest now restores alertness",
                "trigger_desc": "Prolonged wakefulness (hours_awake > 14)",
            },
            {
                "id": "sleep_debt",
                "topic": "Sleep Debt Repair",
                "text": "Sleep extension strategy: go to bed earlier. Avoid sleep debt denial. Weekend recovery without oversleeping. Fatigue is accumulated biology.",
                "trigger_desc": "High cumulative sleep debt",
            },
            {
                "id": "stress_awareness",
                "topic": "Stress vs Activity",
                "text": "Breathing exercises (4-6 breathing). Short walk. Cognitive offloading: write down worries. Differentiate between mental stress and physical load.",
                "trigger_desc": "High heart rate volatility with normal activity",
            },
            {
                "id": "circadian",
                "topic": "Circadian Optimization",
                "text": "Respect biological troughs (1 PM - 3 PM). Use light exposure. Schedule light tasks now. This dip is biological, not personal failure.",
                "trigger_desc": "Biological afternoon slump",
            },
            {
                "id": "physiological",
                "topic": "Physiological Basics",
                "text": "Check hydration. Reduce late-day caffeine. Balanced snacks with protein and fiber. Water intake is critical for energy stability.",
                "trigger_desc": "Elevated HR or afternoon spikes",
            },
        ]
        # 2. Compute once at startup
        self.course_embeddings = self.embedder.encode([c["text"] for c in self.courses])

    def select_course(self, proba, features):
        """
        Rule-Based selection logic with Semantic fallback.
        """
        # Logic Gate 1: Acute Fatigue
        if features.get("hours_awake", 0) > 14:
            return self.courses[0]

        # Logic Gate 2: Sleep Debt
        if features.get("cum_sleep_debt", 0) > 3:
            return self.courses[1]

        # Logic Gate 3: Stress Awareness (stress_cv > threshold)
        if features.get("stress_cv", 0) > 0.15:  # Assuming 0.15 as high stress_cv
            return self.courses[2]

        # Semantic Fallback (if rules don't catch a specific case)
        query = "high fatigue levels and low energy"
        query_vec = self.embedder.encode([query])
        sims = [1 - cosine(query_vec[0], c_vec) for c_vec in self.course_embeddings]
        return self.courses[np.argmax(sims)]

    def get_advice(self, proba, features):
        course = self.select_course(proba, features)
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            litellm.api_key = api_key

            if not api_key:
                print("DEBUG: GROQ_API_KEY is missing from environment variables!")

            response = litellm.completion(
                model="groq/llama-3.1-8b-instant",
                api_key=api_key,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fatigue recovery coach. Use the provided advice to write a supportive 2-line recovery plan.",
                    },
                    {
                        "role": "user",
                        "content": f"User Score: {proba}. Trigger: {course['trigger_desc']}. Expert Advice: {course['text']}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback if API fails: Return raw course text
            print(f"ERROR in LiteLLM/Groq: {str(e)}")
            return f"Coach Insight: {course['text']}"
