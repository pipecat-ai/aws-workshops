def is_off_topic(text):
    """
    Detect if the user's input is off-topic for a health assistant
    """
    # Health-related keywords (simplified list)
    health_keywords = [
        "health", "medical", "doctor", "hospital", "symptom", "treatment",
        "medicine", "disease", "condition", "pain", "diet", "exercise",
        "wellness", "therapy", "diagnosis", "prescription", "vaccine",
        "injury", "recovery", "allergy", "nutrition", "mental health",
        "sleep", "stress", "anxiety", "depression", "blood pressure",
        "heart", "lung", "brain", "stomach", "skin", "bone", "muscle"
    ]
    
    # Explicitly off-topic categories and examples
    off_topic_categories = {
        "entertainment": ["movie", "tv show", "music", "concert", "celebrity", "actor", "singer"],
        "politics": ["politics", "election", "president", "government", "democrat", "republican"],
        "finance": ["stock", "investment", "bitcoin", "crypto", "money", "loan", "bank"],
        "technology": ["computer", "smartphone", "app", "software", "hardware", "coding"],
        "sports": ["football", "basketball", "baseball", "soccer", "tennis", "athlete", "team"],
        "inappropriate": ["joke", "funny", "date", "dating", "relationship", "sex", "illegal"]
    }
    
    # Emergency keywords that should trigger escalation
    emergency_keywords = [
        "emergency", "911", "help me", "chest pain", "heart attack", "stroke", 
        "can't breathe", "difficulty breathing", "severe bleeding", "unconscious",
        "suicide", "kill myself", "overdose", "poisoning", "severe injury",
        "head injury", "seizure", "allergic reaction", "anaphylaxis"
    ]
    
    text_lower = text.lower()
    
    # First check for emergency keywords - these take priority
    for keyword in emergency_keywords:
        if keyword in text_lower:
            return False, None, True  # Not off-topic, but emergency
    
    # Check if any health keyword is in the text
    for keyword in health_keywords:
        if keyword in text_lower:
            return False, None, False  # Not off-topic, not emergency
    
    # Check if any off-topic keyword is in the text
    for category, keywords in off_topic_categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                return True, category, False  # Off-topic, not emergency
    
    # If no health keywords and no explicit off-topic keywords, analyze further
    # This is a simplified approach - in a real application, you might use more sophisticated NLP
    question_starters = ["what", "how", "why", "can", "could", "would", "should", "is", "are", "do"]
    
    # If it's a question but doesn't contain health keywords, it's likely off-topic
    for starter in question_starters:
        if text_lower.startswith(starter):
            return True, "general", False  # Off-topic, not emergency
    
    # Default to not off-topic if we can't determine
    return False, None, False

def get_redirection_response(category=None):
    """
    Generate an appropriate redirection response based on the off-topic category
    """
    general_redirection = (
        "I'm a health assistant designed to help with health-related questions. "
        "Is there something specific about your health or wellness that I can help you with?"
    )
    
    # Specific redirections for certain categories
    specific_redirections = {
        "entertainment": (
            "I'm not able to discuss entertainment topics as I'm designed to assist with health-related questions. "
            "Is there something about your health or wellness that I can help you with instead?"
        ),
        "politics": (
            "I'm not able to discuss political topics as I'm designed to assist with health-related questions. "
            "Is there something about your health or wellness that I can help you with instead?"
        ),
        "finance": (
            "I'm not able to provide financial advice as I'm designed to assist with health-related questions. "
            "Is there something about your health or wellness that I can help you with instead?"
        ),
        "technology": (
            "While technology can certainly impact health, I'm primarily designed to assist with health-related questions. "
            "Is there a specific health concern or wellness topic I can help you with?"
        ),
        "sports": (
            "While physical activity is important for health, I'm not able to discuss sports topics in detail. "
            "I'd be happy to discuss exercise and its health benefits if you're interested."
        ),
        "inappropriate": (
            "I'm designed to provide health information and support. "
            "Please keep our conversation focused on health-related topics that I can assist you with."
        )
    }
    
    if category and category in specific_redirections:
        return specific_redirections[category]
    
    return general_redirection

def get_emergency_response(text):
    """
    Generate an appropriate emergency response based on the detected situation
    """
    # Default emergency response
    general_emergency = (
        "This sounds like a medical emergency. Please call emergency services (911 in the US) immediately. "
        "Do not wait for symptoms to worsen."
    )
    
    # Specific emergency responses
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in ["chest pain", "heart attack", "cardiac"]):
        return (
            "You may be experiencing a cardiac emergency. Call 911 or your local emergency number immediately. "
            "If available and you're not allergic, consider taking aspirin while waiting for help. "
            "Try to remain calm and seated or lying down."
        )
    
    if any(keyword in text_lower for keyword in ["stroke", "face drooping", "arm weakness", "speech difficulty"]):
        return (
            "You may be experiencing stroke symptoms. Call 911 immediately. "
            "Note the time when symptoms started. Do not eat, drink, or take medication. "
            "Lie down with your head slightly elevated and wait for emergency services."
        )
    
    if any(keyword in text_lower for keyword in ["breathing", "breathe", "suffocating", "choking"]):
        return (
            "Difficulty breathing is a serious emergency. Call 911 immediately. "
            "Try to remain calm, loosen any tight clothing, and sit upright if possible. "
            "If you have prescribed rescue medications like an inhaler, use them as directed."
        )
    
    if any(keyword in text_lower for keyword in ["suicide", "kill myself", "end my life", "don't want to live"]):
        return (
            "I'm concerned about what you're saying. Please call the National Suicide Prevention Lifeline at 988 or 1-800-273-8255 immediately. "
            "They have trained counselors available 24/7. You can also text HOME to 741741 to reach the Crisis Text Line. "
            "Please reach out for help - you're not alone."
        )
    
    if any(keyword in text_lower for keyword in ["bleeding", "blood", "hemorrhage"]):
        return (
            "For severe bleeding, call 911 immediately. Apply direct pressure to the wound with a clean cloth or bandage. "
            "If possible, elevate the injured area above the heart. Do not remove the cloth if it becomes soaked - add more on top."
        )
    
    return general_emergency
