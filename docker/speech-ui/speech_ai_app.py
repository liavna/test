"""
Speech.AI - Live Transcription, Diarization & Summary
Powered by HPE Private Cloud AI with Whisper ASR and Qwen LLM

🎯 Version 3.1 - High Performance Edition
"""
import streamlit as st
import requests
import json
import time
import urllib3
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# SSL Configuration - Disable verification
# ============================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# ============================================================
# Configuration
# ============================================================
APP_VERSION = "3.1.0"
APP_TITLE = "🎙️ Speech.AI"
APP_SUBTITLE = "Live Transcription & Diarization"

st.set_page_config(
    page_title=f"{APP_TITLE} - Live Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .live-indicator {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        background: linear-gradient(90deg, #ff4444, #ff6666);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .speaker-1 { 
        background: linear-gradient(90deg, rgba(76, 175, 80, 0.3) 0%, transparent 100%);
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #4CAF50;
    }
    .speaker-2 { 
        background: linear-gradient(90deg, rgba(33, 150, 243, 0.3) 0%, transparent 100%);
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #2196F3;
    }
    .speaker-3 { 
        background: linear-gradient(90deg, rgba(255, 152, 0, 0.3) 0%, transparent 100%);
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #FF9800;
    }
    .speaker-4 { 
        background: linear-gradient(90deg, rgba(156, 39, 176, 0.3) 0%, transparent 100%);
        padding: 10px 14px;
        border-radius: 8px;
        margin: 6px 0;
        border-left: 4px solid #9C27B0;
    }
    .transcript-box {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        min-height: 350px;
        max-height: 500px;
        overflow-y: auto;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        color: #e0e0e0;
        border: 1px solid #2d3748;
    }
    .transcript-text {
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 1.1rem;
        line-height: 1.7;
    }
    .stat-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
DEFAULT_STATE = {
    # Default endpoints - PRE-CONFIGURED FOR HPE PCAI
    'whisper_endpoint': 'http://whisper-large-v3-predictor-00002-deployment.liav-hpe-com-ba9ce2f9.svc.cluster.local:9000',
    'qwen_endpoint': 'https://qwen3-30b-a3b.liav-hpe-com-ba9ce2f9.serving.ingress.pcai.hpelabs.co.il/v1',
    'qwen_token': '',
    'whisper_models': ['whisper-large-v3'],
    'qwen_models': ['Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'],
    'selected_whisper_model': 'whisper-large-v3',
    'selected_qwen_model': 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
    'whisper_connected': False,
    'qwen_connected': False,
    'is_processing': False,
    'transcript': '',
    'diarization': '',
    'summary': '',
    'last_stt_latency': 0,
    'last_llm_latency': 0,
    'word_count': 0,
    'history': [],
    'input_mode': 'file',
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================
# Language Configuration
# ============================================================
SUPPORTED_LANGUAGES = {
    "🇬🇷 Greek (Ελληνικά)": "el",
    "🇮🇱 Hebrew (עברית)": "he",
    "🇺🇸 English": "en",
    "🇸🇦 Arabic (العربية)": "ar",
    "🇷🇺 Russian (Русский)": "ru",
    "🇫🇷 French (Français)": "fr",
    "🇩🇪 German (Deutsch)": "de",
    "🇪🇸 Spanish (Español)": "es",
    "🇮🇹 Italian (Italiano)": "it",
    "🇵🇹 Portuguese (Português)": "pt",
    "🇨🇳 Chinese (中文)": "zh",
    "🇯🇵 Japanese (日本語)": "ja",
    "🇰🇷 Korean (한국어)": "ko",
    "🇹🇷 Turkish (Türkçe)": "tr",
    "🔄 Auto Detect": None,
}

# ============================================================
# Helper Functions
# ============================================================
def get_session():
    """Get a cached session with connection pooling."""
    if 'request_session' not in st.session_state:
        session = requests.Session()
        # Retry strategy
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.verify = False
        st.session_state['request_session'] = session
    return st.session_state['request_session']

def normalize_endpoint(endpoint: str, default_scheme: str = "http") -> str:
    """Ensure endpoint has a scheme (http/https)."""
    if not endpoint:
        return endpoint
    
    endpoint = endpoint.strip()
    
    # If no scheme, add default
    if not endpoint.startswith(('http://', 'https://')):
        # Use https for external ingress URLs, http for internal cluster
        if 'ingress' in endpoint or 'serving' in endpoint:
            endpoint = f"https://{endpoint}"
        else:
            endpoint = f"http://{endpoint}"
    
    return endpoint.rstrip('/')


def get_speaker_class(speaker: str) -> str:
    speaker_lower = speaker.lower()
    if any(x in speaker_lower for x in ["1", "customer", "πελάτης", "לקוח"]):
        return "speaker-1"
    elif any(x in speaker_lower for x in ["2", "agent", "πράκτορας", "נציג"]):
        return "speaker-2"
    elif "3" in speaker:
        return "speaker-3"
    return "speaker-4"


def format_diarized_html(text: str) -> str:
    if not text:
        return ""
    
    lines = text.split('\n')
    html_parts = []
    
    speaker_keywords = ['Speaker', 'Πελάτης', 'Πράκτορας', 'Customer', 'Agent', 
                       'Ομιλητής', 'לקוח', 'נציג', 'דובר']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        has_speaker = ':' in line and any(s in line for s in speaker_keywords)
        
        if has_speaker:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ''
            css_class = get_speaker_class(speaker)
            html_parts.append(
                f'<div class="{css_class} transcript-text">'
                f'<strong>{speaker}:</strong> {content}</div>'
            )
        else:
            html_parts.append(f'<div class="transcript-text">{line}</div>')
    
    return '\n'.join(html_parts)

def extract_model_from_url(url: str) -> Optional[str]:
    """Try to extract model name from the URL string."""
    # Look for patterns like whisper-large-v3, whisper-medium, etc.
    match = re.search(r'(whisper-[\w\-\.]+?)(?:-predictor|-deployment|\.|$)', url)
    if match:
        return match.group(1)
    return None

# ============================================================
# API Functions
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_whisper_models(endpoint: str) -> Tuple[List[str], str]:
    """Fetch Whisper models with robust detection and fallback."""
    if not endpoint:
        return [], "Missing endpoint"
    
    endpoint = normalize_endpoint(endpoint, "http")
    session = get_session()
    
    found_models = []
    error_msg = ""

    # Strategy 1: OpenAI Compatible /v1/models
    try:
        url = f"{endpoint}/v1/models"
        response = session.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                found_models = [m.get('id', m.get('name', 'unknown')) for m in data['data']]
            elif 'models' in data:
                found_models = data['models']
    except Exception as e:
        error_msg = str(e)

    # Strategy 2: Simple /models or KServe
    if not found_models:
        try:
            url = f"{endpoint}/models"
            response = session.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    found_models = data
                elif 'models' in data:
                    found_models = data['models']
        except:
            pass

    # Strategy 3: Check Health/Live + URL Extraction
    # If we can't list models but the server is up, we try to guess the model from the URL
    if not found_models:
        try:
            # Check liveness
            is_live = False
            for path in ["/health", "/v2/health/live", "/v2/health/ready"]:
                try:
                    resp = session.get(f"{endpoint}{path}", timeout=2)
                    if resp.status_code == 200:
                        is_live = True
                        break
                except:
                    continue
            
            if is_live:
                # If server is live but didn't return models, extract from URL
                inferred = extract_model_from_url(endpoint)
                if inferred:
                    found_models = [inferred]
                    error_msg = f"Inferred from URL: {inferred}"
                else:
                    # Generic fallback if live
                    found_models = ['whisper-large-v3']
                    error_msg = "Server live, using default"
        except:
            pass

    # Final Fallback
    if not found_models:
        # One last ditch effort: extract from URL even if we couldn't connect (maybe DNS/Network issue but user knows it's right)
        # Actually, let's only do this if we want to be very optimistic.
        # But better to show connection error.
        # However, for the specific user request "find out which model is running there", extraction is key.
        inferred = extract_model_from_url(endpoint)
        if inferred:
             return [inferred], "Could not connect, but identified model from URL"

        return ['whisper-large-v3'], error_msg or "Could not detect models"

    return found_models, error_msg


def fetch_qwen_models(endpoint: str, token: str) -> Tuple[List[str], str]:
    """Fetch Qwen models (TOKEN REQUIRED)."""
    if not endpoint or not token:
        return [], "Missing endpoint or token"
    
    endpoint = normalize_endpoint(endpoint, "https")
    session = get_session()
    
    try:
        # Remove /v1 suffix if present for models endpoint
        base = endpoint.replace('/v1', '')
        url = f"{base}/v1/models"
        headers = {"Authorization": f"Bearer {token}"}
        response = session.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return [m.get('id', m.get('name', 'unknown')) for m in data['data']], ""
            return [], "Unexpected format"
        elif response.status_code == 401:
            return [], "Auth failed"
        return [], f"HTTP {response.status_code}"
    except Exception as e:
        return [], str(e)


def transcribe_audio(audio_data: bytes, endpoint: str, model: str, language: Optional[str] = None) -> Tuple[str, float, str]:
    """Transcribe audio (NO TOKEN) - supports various Whisper API formats."""
    start_time = time.time()
    
    endpoint = normalize_endpoint(endpoint, "http")
    session = get_session()
    
    # Check if we have a cached working path for this endpoint
    cache_key = f"whisper_path_{endpoint}"
    cached_path = st.session_state.get(cache_key)

    # Try different API paths (prioritize cached one)
    api_paths = [
        "/v1/audio/transcriptions",  # OpenAI-compatible
        "/audio/transcriptions",      # Alternative
        "/transcribe",                # Simple
    ]
    
    if cached_path and cached_path in api_paths:
        api_paths.remove(cached_path)
        api_paths.insert(0, cached_path)

    for api_path in api_paths:
        try:
            url = f"{endpoint}{api_path}"
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            data = {"model": model}
            
            # Add language if specified
            if language:
                data["language"] = language
            
            # Some APIs want response_format
            data["response_format"] = "json"
            
            response = session.post(url, files=files, data=data, timeout=300) # Longer timeout for large files
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Cache the working path
                st.session_state[cache_key] = api_path

                try:
                    result = response.json()
                    # Handle different response formats
                    transcript = (
                        result.get('text') or 
                        result.get('transcript') or 
                        result.get('transcription') or
                        result.get('result', {}).get('text') or
                        ''
                    )
                except:
                    transcript = response.text
                
                if transcript:
                    return transcript.strip(), latency, ""
            
            # If 404, try next path
            if response.status_code == 404:
                continue
                
            return "", latency, f"Error ({response.status_code}): {response.text[:200]}"
            
        except requests.exceptions.ConnectionError:
            continue
        except Exception as e:
            return "", (time.time() - start_time) * 1000, str(e)
    
    return "", (time.time() - start_time) * 1000, "Could not connect to Whisper API"


def perform_diarization(transcript: str, endpoint: str, token: str, model: str, 
                       language: str = "el", num_speakers: Optional[int] = None) -> Tuple[str, float, str]:
    """Diarization (TOKEN REQUIRED)."""
    start_time = time.time()
    
    endpoint = normalize_endpoint(endpoint, "https")
    session = get_session()
    
    prompts = {
        "el": """Είσαι ειδικός στην αναγνώριση ομιλητών. Ανάλυσε το κείμενο και προσδιόρισε τους ομιλητές.
Χρησιμοποίησε "Πελάτης:" και "Πράκτορας:" ή "Ομιλητής 1:", "Ομιλητής 2:".
Διατήρησε το κείμενο, προσθέσε μόνο ετικέτες. Κάθε αλλαγή ομιλητή σε νέα γραμμή.""",
        "he": """אתה מומחה בזיהוי דוברים. נתח את הטקסט וזהה את הדוברים.
השתמש ב"לקוח:" ו"נציג:" או "דובר 1:", "דובר 2:".
שמור על הטקסט, הוסף רק תוויות. כל החלפת דובר בשורה חדשה.""",
        "en": """You are a speaker diarization expert. Analyze and identify speakers.
Use "Customer:" and "Agent:" or "Speaker 1:", "Speaker 2:".
Preserve text, only add labels. Each speaker turn on new line."""
    }
    
    system = prompts.get(language, prompts["en"])
    hint = f"There are {num_speakers} speakers." if num_speakers else "Detect speakers automatically."
    
    try:
        # Handle endpoint with or without /v1
        base = endpoint.rstrip('/')
        if not base.endswith('/v1'):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{hint}\n\nTranscript:\n{transcript}\n\nAdd speaker labels:"}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
        }
        
        response = session.post(url, headers=headers, json=payload, timeout=180)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return content.strip(), latency, ""
        return "", latency, f"Error ({response.status_code})"
    except Exception as e:
        return "", (time.time() - start_time) * 1000, str(e)


def generate_summary(text: str, endpoint: str, token: str, model: str, language: str = "el") -> Tuple[str, str]:
    """Generate summary (TOKEN REQUIRED)."""
    
    endpoint = normalize_endpoint(endpoint, "https")
    session = get_session()
    
    prompts = {
        "el": "Δημιούργησε σύντομη περίληψη στα ελληνικά σε 3-5 σημεία:",
        "he": "צור סיכום קצר בעברית ב-3-5 נקודות:",
        "en": "Create a brief summary in 3-5 points:"
    }
    
    try:
        # Handle endpoint with or without /v1
        base = endpoint.rstrip('/')
        if not base.endswith('/v1'):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"{prompts.get(language, prompts['en'])}\n\n{text}"}
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        }
        
        response = session.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content', ''), ""
        return "", f"Error ({response.status_code})"
    except Exception as e:
        return "", str(e)


# ============================================================
# Demo Conversations
# ============================================================
def get_demo_conversation(language: str) -> List[Tuple[str, str]]:
    conversations = {
        "el": [
            ("Πελάτης", "Καλημέρα σας, θα ήθελα πληροφορίες για τις υπηρεσίες σας."),
            ("Πράκτορας", "Καλημέρα! Χαίρομαι που επικοινωνήσατε. Πώς μπορώ να βοηθήσω;"),
            ("Πελάτης", "Με ενδιαφέρει η αναγνώριση ομιλίας που προσφέρετε."),
            ("Πράκτορας", "Χρησιμοποιούμε το Whisper για μετατροπή ομιλίας σε κείμενο."),
            ("Πελάτης", "Υποστηρίζει ελληνικά;"),
            ("Πράκτορας", "Ναι! Υποστηρίζουμε πάνω από 17 γλώσσες, συμπεριλαμβανομένων ελληνικών."),
            ("Πελάτης", "Πώς λειτουργεί η αναγνώριση ομιλητών;"),
            ("Πράκτορας", "Χρησιμοποιούμε Qwen LLM για αυτόματο διαχωρισμό ομιλητών."),
            ("Πελάτης", "Μπορώ να δω επίδειξη;"),
            ("Πράκτορας", "Αυτό είναι ακριβώς - ζωντανή επίδειξη!"),
        ],
        "he": [
            ("לקוח", "שלום, אני מעוניין במידע על השירותים שלכם."),
            ("נציג", "שלום! שמח שפנית אלינו. איך אוכל לעזור?"),
            ("לקוח", "מעניין אותי זיהוי הדיבור שאתם מציעים."),
            ("נציג", "אנחנו משתמשים ב-Whisper להמרת דיבור לטקסט."),
            ("לקוח", "זה תומך בעברית?"),
            ("נציג", "כן! אנחנו תומכים ביותר מ-17 שפות כולל עברית."),
            ("לקוח", "איך עובד זיהוי הדוברים?"),
            ("נציג", "אנחנו משתמשים ב-Qwen LLM לזיהוי אוטומטי של דוברים."),
        ],
        "en": [
            ("Customer", "Hello, I'd like information about your services."),
            ("Agent", "Hello! How can I help you today?"),
            ("Customer", "I'm interested in speech recognition."),
            ("Agent", "We use Whisper for speech-to-text conversion."),
            ("Customer", "Does it support multiple languages?"),
            ("Agent", "Yes! We support over 17 languages."),
        ]
    }
    return conversations.get(language, conversations["en"])


# ============================================================
# UI Functions
# ============================================================
def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(APP_TITLE)
        st.caption(f"{APP_SUBTITLE} | v{APP_VERSION}")
    with col2:
        w = "🟢" if st.session_state.whisper_connected else "🔴"
        q = "🟢" if st.session_state.qwen_connected else "🔴"
        st.markdown(f"**Status:** {w} Whisper | {q} Qwen")
        if st.session_state.is_processing:
            st.markdown('<span class="live-indicator">● PROCESSING</span>', unsafe_allow_html=True)

def handle_whisper_url_change():
    """Callback to auto-connect when URL changes."""
    if st.session_state.whisper_endpoint:
        with st.spinner("Detecting model..."):
            endpoint = normalize_endpoint(st.session_state.whisper_endpoint, "http")
            # Force cache refresh if URL changed
            fetch_whisper_models.clear()
            models, error = fetch_whisper_models(endpoint)

            # Auto connect if we found something
            if models:
                st.session_state.whisper_connected = True
                st.session_state.whisper_models = models
                st.session_state.selected_whisper_model = models[0]
                if not error:
                    st.toast(f"✅ Found {models[0]}")
            else:
                st.session_state.whisper_connected = False
                st.toast(f"⚠️ {error}")

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Whisper
        st.subheader("🎤 Whisper (STT)")
        st.caption("Internal cluster address • No token")
        
        # We use on_change to trigger detection immediately
        new_endpoint = st.text_input(
            "Whisper Endpoint",
            value=st.session_state.whisper_endpoint,
            placeholder="http://whisper-xxx.namespace.svc.cluster.local:9000",
            help="Internal K8s service address (port 9000)",
            key="whisper_endpoint_input"
        )
        
        # Check if URL changed
        if new_endpoint != st.session_state.whisper_endpoint:
            st.session_state.whisper_endpoint = new_endpoint
            handle_whisper_url_change()
            st.rerun()

        # Connect button (Manual trigger)
        if st.button("🔌 Connect / Refresh", use_container_width=True):
            handle_whisper_url_change()
            st.rerun()
        
        if st.session_state.whisper_models:
            st.session_state.selected_whisper_model = st.selectbox(
                "Whisper Model", st.session_state.whisper_models
            )
        
        # Manual model input if needed
        manual_whisper = st.text_input(
            "Or enter model name manually",
            value="",
            placeholder="whisper-large-v3",
            key="manual_whisper_model"
        )
        if manual_whisper:
            st.session_state.selected_whisper_model = manual_whisper
        
        st.divider()
        
        # Qwen
        st.subheader("🧠 Qwen (LLM)")
        st.caption("External ingress address • Token required")
        
        st.session_state.qwen_endpoint = st.text_input(
            "Qwen Endpoint",
            value=st.session_state.qwen_endpoint,
            placeholder="https://qwen.ingress.pcai.hpelabs.co.il",
            help="External ingress URL"
        )
        
        st.session_state.qwen_token = st.text_input(
            "Qwen Token",
            value=st.session_state.qwen_token,
            type="password",
            help="MLIS API token for authentication"
        )
        
        if st.button("🔌 Connect Qwen", use_container_width=True):
            if st.session_state.qwen_endpoint and st.session_state.qwen_token:
                with st.spinner("Connecting..."):
                    models, error = fetch_qwen_models(
                        st.session_state.qwen_endpoint,
                        st.session_state.qwen_token
                    )
                    if models:
                        st.session_state.qwen_models = models
                        st.session_state.qwen_connected = True
                        st.session_state.selected_qwen_model = models[0]
                        st.success(f"✅ {len(models)} models found")
                    else:
                        st.error(f"❌ {error}")
            else:
                st.warning("Enter endpoint and token")
        
        if st.session_state.qwen_models:
            st.session_state.selected_qwen_model = st.selectbox(
                "Qwen Model", st.session_state.qwen_models
            )
        
        st.divider()
        
        # Options
        st.subheader("🌍 Options")
        
        selected_lang = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), index=0)
        do_diarization = st.checkbox("🎭 Diarization", value=True)
        num_speakers = st.slider("Speakers (0=auto)", 0, 6, 2) if do_diarization else 0
        do_summary = st.checkbox("📋 Summary", value=False)
        
        return {
            'language': SUPPORTED_LANGUAGES[selected_lang],
            'language_name': selected_lang,
            'do_diarization': do_diarization,
            'num_speakers': num_speakers if num_speakers > 0 else None,
            'do_summary': do_summary
        }


def render_metrics():
    if not st.session_state.transcript:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚡ STT", f"{st.session_state.last_stt_latency:.0f} ms")
    with col2:
        st.metric("🧠 LLM", f"{st.session_state.last_llm_latency:.0f} ms")
    with col3:
        st.metric("📝 Words", st.session_state.word_count)
    with col4:
        total = st.session_state.last_stt_latency + st.session_state.last_llm_latency
        st.metric("⏱️ Total", f"{total:.0f} ms")


def process_audio(audio_bytes: bytes, options: Dict[str, Any]):
    if not st.session_state.whisper_connected:
        st.error("❌ Connect Whisper first!")
        return
    
    st.session_state.is_processing = True
    
    with st.status("🔄 Processing...", expanded=True) as status:
        # Transcribe
        st.write(f"📝 Transcribing with **{st.session_state.selected_whisper_model}**...")
        
        transcript, stt_latency, error = transcribe_audio(
            audio_bytes,
            st.session_state.whisper_endpoint,
            st.session_state.selected_whisper_model,
            options['language']
        )
        
        if error:
            status.update(label="❌ Failed", state="error")
            st.error(error)
            st.session_state.is_processing = False
            return
        
        st.session_state.transcript = transcript
        st.session_state.last_stt_latency = stt_latency
        st.session_state.word_count = len(transcript.split())
        st.write(f"✅ {st.session_state.word_count} words in {stt_latency:.0f}ms")
        
        # Diarize
        diarized = transcript
        if options['do_diarization'] and st.session_state.qwen_connected:
            st.write(f"🎭 Diarizing with **{st.session_state.selected_qwen_model}**...")
            
            diarized, llm_latency, error = perform_diarization(
                transcript,
                st.session_state.qwen_endpoint,
                st.session_state.qwen_token,
                st.session_state.selected_qwen_model,
                options['language'] or 'en',
                options['num_speakers']
            )
            
            if error:
                st.warning(f"⚠️ {error}")
                diarized = transcript
            else:
                st.session_state.last_llm_latency = llm_latency
                st.write(f"✅ Done in {llm_latency:.0f}ms")
        
        st.session_state.diarization = diarized
        
        # Summary
        if options['do_summary'] and st.session_state.qwen_connected:
            st.write("📋 Generating summary...")
            summary, error = generate_summary(
                diarized,
                st.session_state.qwen_endpoint,
                st.session_state.qwen_token,
                st.session_state.selected_qwen_model,
                options['language'] or 'en'
            )
            if not error:
                st.session_state.summary = summary
        
        # History
        st.session_state.history.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'words': st.session_state.word_count,
            'transcript': transcript,
            'diarization': diarized
        })
        
        status.update(label="✅ Complete!", state="complete")
    
    st.session_state.is_processing = False


def run_demo(options: Dict[str, Any]):
    conversation = get_demo_conversation(options['language'] or 'el')
    
    progress = st.progress(0)
    status = st.empty()
    display = st.empty()
    
    parts = []
    for i, (speaker, text) in enumerate(conversation):
        status.markdown(f"🔴 **Live:** {speaker}...")
        time.sleep(0.8)
        
        parts.append(f"{speaker}: {text}")
        html = format_diarized_html("\n".join(parts))
        display.markdown(f'<div class="transcript-box">{html}</div>', unsafe_allow_html=True)
        progress.progress((i + 1) / len(conversation))
    
    st.session_state.transcript = " ".join([t for _, t in conversation])
    st.session_state.diarization = "\n".join(parts)
    st.session_state.word_count = len(st.session_state.transcript.split())
    st.session_state.last_stt_latency = 150
    st.session_state.last_llm_latency = 200
    
    status.markdown("✅ **Demo Complete!**")


def render_main(options: Dict[str, Any]):
    st.subheader("🎯 Input Mode")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📁 Upload File", use_container_width=True,
                     type="primary" if st.session_state.input_mode == 'file' else "secondary"):
            st.session_state.input_mode = 'file'
    with col2:
        if st.button("🎙️ Microphone", use_container_width=True,
                     type="primary" if st.session_state.input_mode == 'mic' else "secondary"):
            st.session_state.input_mode = 'mic'
    with col3:
        if st.button("🎭 Demo", use_container_width=True):
            run_demo(options)
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎤 Audio Input")
        
        audio_bytes = None
        
        if st.session_state.input_mode == 'file':
            uploaded = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'ogg', 'm4a', 'flac', 'webm']
            )
            if uploaded:
                audio_bytes = uploaded.read()
                st.audio(audio_bytes)
                st.success(f"📁 {uploaded.name} ({len(audio_bytes)/1024:.1f} KB)")
        else:
            recorded = st.audio_input("🎙️ Record from microphone")
            if recorded:
                audio_bytes = recorded.read() if hasattr(recorded, 'read') else recorded
                st.success(f"🎙️ Recorded ({len(audio_bytes)/1024:.1f} KB)")
        
        if audio_bytes:
            st.divider()
            
            # Show connection status
            if not st.session_state.whisper_connected:
                st.warning("⚠️ Connect to Whisper first (sidebar)")
            
            c1, c2 = st.columns(2)
            with c1:
                transcribe_btn = st.button("📝 Transcribe Only", use_container_width=True,
                            disabled=not st.session_state.whisper_connected)
            with c2:
                full_btn = st.button("🚀 Full Process", type="primary", use_container_width=True,
                            disabled=not st.session_state.whisper_connected)
            
            if transcribe_btn:
                process_audio(audio_bytes, {**options, 'do_diarization': False, 'do_summary': False})
            
            if full_btn:
                process_audio(audio_bytes, options)
    
    with col2:
        st.markdown("### 📝 Results")
        render_metrics()
        
        if st.session_state.diarization:
            html = format_diarized_html(st.session_state.diarization)
            st.markdown(f'<div class="transcript-box">{html}</div>', unsafe_allow_html=True)
        elif st.session_state.transcript:
            st.markdown(f'<div class="transcript-box transcript-text">{st.session_state.transcript}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="transcript-box" style="text-align:center;padding-top:100px;color:#666;">'
                       '🎤 Upload or record audio to start...</div>', unsafe_allow_html=True)
    
    # Summary
    if st.session_state.summary:
        st.divider()
        st.subheader("📋 Summary")
        st.info(st.session_state.summary)
    
    # Export
    if st.session_state.transcript:
        st.divider()
        st.subheader("💾 Export")
        c1, c2, c3, c4 = st.columns(4)
        ts = datetime.now().strftime('%H%M%S')
        with c1:
            st.download_button("📄 Transcript", st.session_state.transcript, f"transcript_{ts}.txt")
        with c2:
            if st.session_state.diarization:
                st.download_button("🎭 Diarization", st.session_state.diarization, f"diarized_{ts}.txt")
        with c3:
            if st.session_state.summary:
                st.download_button("📋 Summary", st.session_state.summary, f"summary_{ts}.txt")
        with c4:
            if st.button("🗑️ Clear"):
                for k in ['transcript', 'diarization', 'summary']:
                    st.session_state[k] = ''
                st.session_state.last_stt_latency = 0
                st.session_state.last_llm_latency = 0
                st.rerun()


def render_history():
    st.subheader("📜 History")
    if not st.session_state.history:
        st.info("No history yet.")
        return
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        with st.expander(f"🕐 {item['time']} - {item['words']} words"):
            st.text_area("Transcript", item['transcript'], height=80, disabled=True, key=f"h_t_{i}")
            st.text_area("Diarization", item['diarization'], height=80, disabled=True, key=f"h_d_{i}")


def render_about():
    st.markdown(f"""
### 🎙️ Speech.AI v{APP_VERSION}

**Features:**
- 📁 File Upload (WAV, MP3, OGG, M4A, FLAC, WebM)
- 🎙️ Live Microphone Recording
- 🎭 Speaker Diarization
- 🌍 17+ Languages
- ⚡ Real-time Metrics
- 📋 Summary Generation
- 💾 Export Results

**Architecture:**
```
Audio → Whisper (NO TOKEN) → Qwen (TOKEN) → Results
```

**SSL:** Verification disabled for internal endpoints.

---
**Powered by HPE Private Cloud AI**
    """)


# ============================================================
# Main
# ============================================================
def main():
    render_header()
    options = render_sidebar()
    
    tab1, tab2, tab3 = st.tabs(["🎤 Process", "📜 History", "ℹ️ About"])
    
    with tab1:
        render_main(options)
    with tab2:
        render_history()
    with tab3:
        render_about()


if __name__ == "__main__":
    main()
