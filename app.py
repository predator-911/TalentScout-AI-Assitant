import streamlit as st
import time
import random
from datetime import datetime
import json
import os
from textblob import TextBlob
from deep_translator import GoogleTranslator
from langdetect import detect
import plotly.graph_objects as go
from transformers import pipeline

# Technical questions bank
TECH_QUESTIONS = {
    "python": [
        "Explain the difference between a list and a tuple in Python.",
        "How would you handle exceptions in Python?",
        "What are decorators in Python and how do they work?",
        "Explain list comprehensions and provide an example.",
        "How does memory management work in Python?",
    ],
    "javascript": [
        "What's the difference between '==' and '===' in JavaScript?",
        "Explain closures in JavaScript with an example.",
        "How does prototypal inheritance work?",
        "What are Promises and how do they differ from callbacks?",
        "Explain event delegation in JavaScript.",
    ],
    "react": [
        "What are React hooks and why were they introduced?",
        "Explain the component lifecycle in React.",
        "What is the virtual DOM and how does it work?",
        "How would you optimize performance in a React application?",
        "Explain the context API and when you would use it.",
    ],
    "java": [
        "What's the difference between an interface and an abstract class?",
        "Explain garbage collection in Java.",
        "What are generics and why are they useful?",
        "How does multithreading work in Java?",
        "What are the key principles of OOP in Java?",
    ],
    "sql": [
        "What's the difference between INNER JOIN and LEFT JOIN?",
        "Explain normalization and when you would use it.",
        "How would you optimize a slow SQL query?",
        "What are indexes and how do they work?",
        "Explain the difference between DELETE and TRUNCATE.",
    ],
    "mongodb": [
        "How does MongoDB store data compared to SQL databases?",
        "Explain sharding in MongoDB.",
        "What are the ACID properties in MongoDB?",
        "How would you design schema for a social media application?",
        "Explain indexing strategies in MongoDB.",
    ],
    "docker": [
        "What's the difference between Docker and virtual machines?",
        "Explain Docker layers and how they work.",
        "How would you persist data in Docker?",
        "Explain Docker networking concepts.",
        "What is Docker Compose and when would you use it?",
    ],
    "aws": [
        "Explain the differences between EC2, ECS, and Lambda.",
        "How would you design a highly available architecture in AWS?",
        "What are the key security best practices in AWS?",
        "Explain the concept of IAM and role-based access.",
        "How does S3 storage work and what are its use cases?",
    ],
    "django": [
        "Explain the MTV architecture in Django.",
        "How does the ORM work in Django?",
        "What are middleware in Django and how are they used?",
        "Explain Django's authentication system.",
        "How would you optimize a Django application for performance?",
    ],
    "nodejs": [
        "How does the event loop work in Node.js?",
        "What's the difference between process.nextTick() and setImmediate()?",
        "How would you handle async operations in Node.js?",
        "Explain the module system in Node.js.",
        "What are streams in Node.js and how would you use them?",
    ],
    "css": [
        "Explain the box model in CSS.",
        "What's the difference between flexbox and grid?",
        "How does CSS specificity work?",
        "Explain CSS positioning (relative, absolute, fixed, sticky).",
        "What are CSS preprocessors and what benefits do they provide?",
    ],
    "html": [
        "What's new in HTML5?",
        "Explain semantic HTML and why it's important.",
        "How do you optimize HTML for accessibility?",
        "What are data attributes and how are they used?",
        "Explain the critical rendering path in browsers.",
    ],
    "devops": [
        "What is CI/CD and how does it benefit development?",
        "Explain infrastructure as code and its benefits.",
        "How would you implement blue/green deployment?",
        "What monitoring tools have you used and why?",
        "How do you approach logging in a microservices architecture?",
    ],
    "git": [
        "Explain the difference between merge and rebase.",
        "How would you fix a bad commit that's already pushed?",
        "What's your branching strategy preference and why?",
        "Explain git hooks and how they can be used.",
        "How do you handle merge conflicts?",
    ],
}

# Default questions for unknown tech stacks
DEFAULT_QUESTIONS = [
    "Can you describe your experience with this technology?",
    "What projects have you worked on using this technology?",
    "What are some challenges you've faced with this technology and how did you overcome them?",
    "How do you stay updated with the latest developments in this field?",
    "Can you explain a complex concept in this technology in simple terms?"
]

# Fallback responses when the bot doesn't understand
FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you please rephrase that?",
    "I didn't quite catch that. Can you elaborate?",
    "I'm having trouble following. Could you clarify what you mean?",
    "I'm sorry, I didn't understand. Let's try a different approach.",
    "I may have missed something. Could you provide more details?"
]

# Exit phrases that trigger the conversation ending
EXIT_PHRASES = ["bye", "goodbye", "exit", "quit", "end", "thank you", "thanks"]

# Supported languages and their codes
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Portuguese": "pt"
}

# Function to initialize session state variables
def initialize_session_state():
    if 'stage' not in st.session_state:
        st.session_state.stage = "greeting"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'candidate_info' not in st.session_state:
        st.session_state.candidate_info = {
            "name": "",
            "email": "",
            "phone": "",
            "experience": "",
            "position": "",
            "location": "",
            "tech_stack": []
        }
    if 'tech_stack_str' not in st.session_state:
        st.session_state.tech_stack_str = ""
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = []
    if 'current_tech' not in st.session_state:
        st.session_state.current_tech = ""
    if 'asked_questions_count' not in st.session_state:
        st.session_state.asked_questions_count = 0
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False
    if 'sentiment_history' not in st.session_state:
        st.session_state.sentiment_history = []
    if 'language' not in st.session_state:
        st.session_state.language = "en"
    if 'language_name' not in st.session_state:
        st.session_state.language_name = "English"
    if 'llm_model' not in st.session_state:
        # Initialize the language model
        try:
            st.session_state.llm_model = pipeline(
                "text-generation", 
                model="distilgpt2", 
                max_length=100
            )
        except:
            st.session_state.llm_model = None
            print("Warning: HuggingFace model could not be loaded. Using predefined questions.")

# Sentiment analysis function using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    # Record sentiment for visualization
    st.session_state.sentiment_history.append((len(st.session_state.messages), sentiment_score))
    
    # Return sentiment category and score
    if sentiment_score > 0.3:
        return "positive", sentiment_score
    elif sentiment_score < -0.3:
        return "negative", sentiment_score
    else:
        return "neutral", sentiment_score

# Function to detect language and translate text
def detect_and_translate(text, target_lang="en"):
    try:
        detected_lang = detect(text)
        
        # If detected language is not target language, translate it
        if detected_lang != target_lang:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated, detected_lang
        else:
            return text, detected_lang
    except:
        # If detection fails, return original text
        return text, "en"

# Function to translate text to user's preferred language
def translate_to_user_language(text, source_lang="en"):
    target_lang = st.session_state.language
    if source_lang != target_lang:
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            return translated
        except:
            return text
    return text

# Function to generate questions using Hugging Face model
def generate_technical_question(tech):
    if st.session_state.llm_model is None:
        return None
    
    prompt = f"Create a challenging technical interview question about {tech} for a software developer position:"
    
    try:
        result = st.session_state.llm_model(prompt)[0]['generated_text']
        # Clean up the result to get just the question
        question = result.split(prompt)[1].strip()
        # If question is too short or incomplete, return None
        if len(question) < 20 or "?" not in question:
            return None
        return question
    except:
        return None

# Function to save chat history to JSON file
def save_chat_history():
    if not os.path.exists("chat_histories"):
        os.makedirs("chat_histories")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_histories/chat_{timestamp}.json"
    
    data = {
        "candidate_info": st.session_state.candidate_info,
        "messages": st.session_state.messages,
        "sentiment_history": st.session_state.sentiment_history
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# Function to add a message to the chat
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    
    # Analyze sentiment if it's a user message
    if role == "user":
        sentiment_category, sentiment_score = analyze_sentiment(content)
        
        # Save sentiment information with the message
        st.session_state.messages[-1]["sentiment"] = {
            "category": sentiment_category,
            "score": sentiment_score
        }

# Function to create sentiment visualization
def display_sentiment_visualization():
    if len(st.session_state.sentiment_history) > 1:
        x = [item[0] for item in st.session_state.sentiment_history]
        y = [item[1] for item in st.session_state.sentiment_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, 
            y=y,
            mode='lines+markers',
            marker=dict(
                size=8,
                color=[get_sentiment_color(score) for score in y],
                colorscale='RdYlGn',
                showscale=False
            ),
            line=dict(
                width=2,
                color='royalblue'
            )
        ))
        
        fig.update_layout(
            title="Candidate Sentiment Throughout Conversation",
            xaxis_title="Message Number",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to get color for sentiment visualization
def get_sentiment_color(score):
    if score > 0.3:
        return "green"
    elif score < -0.3:
        return "red"
    else:
        return "yellow"

# Function to display existing messages with animations
def display_chat_history():
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sentiment for user messages if available
            if message["role"] == "user" and "sentiment" in message:
                sentiment = message["sentiment"]
                
                # Use colored indicators for sentiment
                if sentiment["category"] == "positive":
                    st.markdown("<span style='color:green; font-size:12px;'>😊 Positive</span>", unsafe_allow_html=True)
                elif sentiment["category"] == "negative":
                    st.markdown("<span style='color:red; font-size:12px;'>😞 Negative</span>", unsafe_allow_html=True)

# Function to handle user input based on current stage
def process_user_input(user_input):
    # First, check if we need language detection and translation
    translated_input, detected_lang = detect_and_translate(user_input, "en")
    
    # If a different language was detected, update the user's preferred language
    if detected_lang != "en" and detected_lang in SUPPORTED_LANGUAGES.values():
        st.session_state.language = detected_lang
        # Find language name from code
        for name, code in SUPPORTED_LANGUAGES.items():
            if code == detected_lang:
                st.session_state.language_name = name
                break
    
    # Use the translated input for processing
    working_input = translated_input
    
    # Add the original user input to messages
    add_message("user", user_input)
    
    # Check for exit phrases
    if any(phrase in working_input.lower() for phrase in EXIT_PHRASES) and st.session_state.stage != "greeting":
        handle_exit()
        return
    
    # Process based on current stage
    if st.session_state.stage == "greeting":
        st.session_state.stage = "name"
        ask_name()
    
    elif st.session_state.stage == "name":
        st.session_state.candidate_info["name"] = working_input
        st.session_state.stage = "email"
        ask_email()
    
    elif st.session_state.stage == "email":
        st.session_state.candidate_info["email"] = working_input
        st.session_state.stage = "phone"
        ask_phone()
    
    elif st.session_state.stage == "phone":
        st.session_state.candidate_info["phone"] = working_input
        st.session_state.stage = "experience"
        ask_experience()
    
    elif st.session_state.stage == "experience":
        st.session_state.candidate_info["experience"] = working_input
        st.session_state.stage = "position"
        ask_position()
    
    elif st.session_state.stage == "position":
        st.session_state.candidate_info["position"] = working_input
        st.session_state.stage = "location"
        ask_location()
    
    elif st.session_state.stage == "location":
        st.session_state.candidate_info["location"] = working_input
        st.session_state.stage = "tech_stack"
        ask_tech_stack()
    
    elif st.session_state.stage == "tech_stack":
        st.session_state.tech_stack_str = working_input
        
        # Process the tech stack string into a list
        techs = [tech.strip().lower() for tech in working_input.split(',')]
        st.session_state.candidate_info["tech_stack"] = techs
        
        # Find valid technologies in our question bank
        valid_techs = [tech for tech in techs if tech in TECH_QUESTIONS]
        
        if valid_techs:
            st.session_state.stage = "technical_questions"
            st.session_state.current_tech = valid_techs[0]
            ask_technical_questions()
        else:
            handle_unknown_tech_stack()
    
    elif st.session_state.stage == "technical_questions":
        st.session_state.asked_questions_count += 1
        
        # If we've asked enough questions about the current technology
        if st.session_state.asked_questions_count >= 3:
            # Move to the next technology if there are more
            techs = st.session_state.candidate_info["tech_stack"]
            current_index = techs.index(st.session_state.current_tech)
            
            if current_index + 1 < len(techs):
                st.session_state.current_tech = techs[current_index + 1]
                st.session_state.asked_questions_count = 0
                ask_technical_questions()
            else:
                # No more technologies to ask about
                st.session_state.stage = "wrap_up"
                wrap_up_interview()
        else:
            # Ask another question about the current technology
            ask_technical_questions()
    
    elif st.session_state.stage == "wrap_up":
        handle_exit()
    
    else:
        # Fallback for unexpected stage
        handle_fallback()

# Stage-specific functions

def greet():
    greeting = """
    👋 Hello! I'm the TalentScout AI Assistant.
    
    I'll be helping you through the initial screening process for your job application.
    I'll ask you a series of questions to learn more about you and your technical skills.
    
    Let's get started! How are you doing today?
    """
    # Translate greeting if needed
    if st.session_state.language != "en":
        greeting = translate_to_user_language(greeting)
    
    add_message("assistant", greeting)

def ask_name():
    message = "First, could you please tell me your full name?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_email():
    message = f"Nice to meet you, {st.session_state.candidate_info['name']}! Could you please provide your email address?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_phone():
    message = "Great! Now, could you share your phone number?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_experience():
    message = "How many years of experience do you have in your field?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_position():
    message = f"Thanks! What position(s) are you interested in applying for at TalentScout?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_location():
    message = "What is your current location?"
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_tech_stack():
    message = "Please list the technologies you're proficient in, separated by commas (e.g., Python, JavaScript, React, MongoDB)."
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    add_message("assistant", message)

def ask_technical_questions():
    current_tech = st.session_state.current_tech
    
    # Try to generate a question using the LLM first
    llm_question = None
    if st.session_state.llm_model is not None:
        llm_question = generate_technical_question(current_tech)
    
    # If LLM generated a valid question, use it
    if llm_question:
        question = llm_question
        st.session_state.questions_asked.append(question)
        message = f"About {current_tech.capitalize()}: {question}"
    else:
        # Fall back to predefined questions
        if current_tech.lower() in TECH_QUESTIONS:
            questions = TECH_QUESTIONS[current_tech.lower()]
            
            # Filter out already asked questions
            available_questions = [q for q in questions if q not in st.session_state.questions_asked]
            
            if available_questions:
                # Select a random question
                question = random.choice(available_questions)
                st.session_state.questions_asked.append(question)
                
                message = f"About {current_tech.capitalize()}: {question}"
            else:
                # All questions for this tech have been asked
                techs = st.session_state.candidate_info["tech_stack"]
                current_index = techs.index(current_tech)
                
                if current_index + 1 < len(techs):
                    # Move to the next technology
                    st.session_state.current_tech = techs[current_index + 1]
                    st.session_state.asked_questions_count = 0
                    ask_technical_questions()
                    return
                else:
                    # No more technologies to ask about
                    st.session_state.stage = "wrap_up"
                    wrap_up_interview()
                    return
        else:
            # Use default questions for unknown tech stacks
            question = random.choice(DEFAULT_QUESTIONS)
            message = f"About {current_tech.capitalize()}: {question}"
    
    # Translate the question if needed
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    
    add_message("assistant", message)

def handle_unknown_tech_stack():
    message = "I don't have specific technical questions for the technologies you've mentioned. Let's have a more general discussion about your skills."
    
    # Translate if needed
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    
    add_message("assistant", message)
    
    # Ask a general technical question
    question = "Can you describe your technical background and the projects you've worked on?"
    
    if st.session_state.language != "en":
        question = translate_to_user_language(question)
    
    add_message("assistant", question)
    
    st.session_state.stage = "wrap_up"

def wrap_up_interview():
    message = f"""
    Thank you for answering the technical questions, {st.session_state.candidate_info['name']}!
    
    Based on our conversation, I have a good understanding of your background and technical skills.
    
    Is there anything else you'd like to share about yourself or do you have any questions about the position?
    """
    
    # Translate if needed
    if st.session_state.language != "en":
        message = translate_to_user_language(message)
    
    add_message("assistant", message)

def handle_exit():
    if not st.session_state.conversation_ended:
        farewell = f"""
        Thank you for taking the time to chat with me today, {st.session_state.candidate_info['name']}!
        
        Our team at TalentScout will review your profile, and we'll be in touch via email ({st.session_state.candidate_info['email']}) or phone ({st.session_state.candidate_info['phone']}) within the next 3-5 business days.
        
        Have a great day!
        """
        
        # Translate farewell if needed
        if st.session_state.language != "en":
            farewell = translate_to_user_language(farewell)
        
        add_message("assistant", farewell)
        st.session_state.conversation_ended = True
        
        # Save the chat history
        save_chat_history()

def handle_fallback():
    fallback = random.choice(FALLBACK_RESPONSES)
    
    # Translate fallback if needed
    if st.session_state.language != "en":
        fallback = translate_to_user_language(fallback)
    
    add_message("assistant", fallback)

# Custom CSS for a more polished UI
def load_css():
    st.markdown("""
    <style>
    /* Custom styles for the chat interface */
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    
    /* Adding a pulsing effect to the chat avatar */
    @keyframes pulseAnimation {
        0% { box-shadow: 0 0 0 0 rgba(30, 144, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(30, 144, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(30, 144, 255, 0); }
    }
    
    .stChatMessage div[data-testid="stChatMessageAvatar"] {
        animation: pulseAnimation 2s infinite;
        border-radius: 50%;
    }
    
    /* Custom styling for the messages */
    .stChatMessage {
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom header styling */
    h1, h2, h3 {
        color: #1E90FF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        background-color: #1E90FF;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0066CC;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Animation for new messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage:last-child {
        animation: fadeIn 0.5s ease;
    }
    
    /* Custom selectbox styling */
    .stSelectbox>div>div {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Main app function
def main():
    st.set_page_config(
        page_title="TalentScout AI Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    # Load custom CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Display header with animation
    st.markdown("""
    <div style="text-align: center; animation: fadeIn 1s ease;">
        <h1>🤖 TalentScout AI Assistant</h1>
        <h3>Your AI-powered hiring companion</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector in the sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Language selector
        selected_language = st.selectbox(
            "Select Your Language",
            list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.language_name) 
                if st.session_state.language_name in SUPPORTED_LANGUAGES.keys() else 0
        )
        
        # Update language if changed
        if SUPPORTED_LANGUAGES[selected_language] != st.session_state.language:
            st.session_state.language = SUPPORTED_LANGUAGES[selected_language]
            st.session_state.language_name = selected_language
        
        # Display sentiment analysis if conversation has progressed
        if len(st.session_state.sentiment_history) > 1:
            st.header("Conversation Analysis")
            display_sentiment_visualization()
        
        # About section
        st.header("About")
        st.markdown("""
        The TalentScout AI Assistant helps streamline the initial candidate screening process. 
        It collects basic information and asks relevant technical questions based on the candidate's declared skill set.
        
        **Features:**
        - Interactive chat interface
        - Multilingual support
        - Sentiment analysis
        - Dynamic technical questions
        """)
    
    # Main container with chat interface
    with st.container():
        # Display chat interface
        display_chat_history()
        
        # Initialize with greeting if it's the first interaction
        if st.session_state.stage == "greeting" and not st.session_state.messages:
            greet()
        
        # User input
        if not st.session_state.conversation_ended:
            user_input = st.chat_input("Type your message here...")
            if user_input:
                process_user_input(user_input)
                st.rerun()
        
        # Display a restart button if conversation has ended
        if st.session_state.conversation_ended:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Start New Conversation", use_container_width=True):
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

if __name__ == "__main__":
    main()

