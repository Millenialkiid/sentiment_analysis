import streamlit as st
from sentiment_stub import predict_sentiment, load_model_and_preprocessor
import logging

# --- Page Config ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# --- Logging ---
# Configure logging to see errors in the terminal
logging.basicConfig(level=logging.INFO)

# --- Model Loading ---
# Use Streamlit's cache to load the model only once.
@st.cache_resource
def warm_up():
    """
    Loads the model and preprocessor into memory.
    This will run once when the app starts.
    """
    try:
        load_model_and_preprocessor()
        return True
    except FileNotFoundError as e:
        st.error(f"ERROR: Model file not found. {e}")
        st.error("Please run your training script (or train.bat) to create the model file.")
        return False
    except ImportError as e:
        st.error(f"ERROR: Could not import project code. {e}")
        st.error("Please ensure your 'src' folder and venv are set up correctly.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred on startup: {e}")
        return False

# Try to load the model on startup
model_ready = warm_up()

# --- App UI ---
st.title("🎬 IMDB Sentiment Analyzer")
st.markdown("Enter a movie review below to analyze its sentiment (Positive or Negative).")

# Create the text area
review_text = st.text_area("Enter a movie review:", height=150, placeholder="This movie was fantastic!...")

# Create the button, but disable it if the model failed to load
analyze_button = st.button("Analyze Sentiment", type="primary", disabled=not model_ready)

if analyze_button:
    if not review_text.strip():
        # Check if the text area is empty
        st.warning("Please enter a review to analyze.")
    else:
        # Show a spinner while predicting
        with st.spinner("Analyzing..."):
            try:
                # Call the prediction function from our bridge file
                result = predict_sentiment(review_text)
                
                if result.get("sentiment") == "Error":
                    # Handle prediction errors
                    st.error(f"An error occurred during prediction: {result.get('error')}")
                else:
                    # Display the results
                    sentiment = result.get("sentiment")
                    confidence = result.get("confidence", 0.0)
                    
                    if sentiment == "Positive":
                        st.success(f"**Sentiment: {sentiment}** (Confidence: {confidence*100:.1f}%)")
                    else:
                        st.error(f"**Sentiment: {sentiment}** (Confidence: {confidence*100:.1f}%)")
                    
                    # Optional: Show the raw probability
                    # st.write(f"Raw Probability: {result.get('probability', 0.0):.4f}")

            except Exception as e:
                st.error(f"A fatal error occurred: {e}")
                logging.error(f"Fatal error in predict_sentiment: {e}")

elif not model_ready:
    st.error("The application cannot start because the model failed to load.")

