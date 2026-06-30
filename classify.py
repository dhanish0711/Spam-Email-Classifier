import os
import sys
import argparse
import joblib

# Paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def load_classifier():
    """Load the model and vectorizer from the models directory."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print(f"{Colors.RED}{Colors.BOLD}Error:{Colors.END} Model files not found. "
              "Please run training first using: python train.py")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        print(f"{Colors.RED}{Colors.BOLD}Error:{Colors.END} Failed to load model artifacts: {e}")
        sys.exit(1)

def classify_text(text, model, vectorizer):
    """Classify a single text string as Spam or Not Spam."""
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    
    classes = list(model.classes_)
    probs = model.predict_proba(text_vectorized)[0]
    
    pred_idx = classes.index(prediction)
    confidence = probs[pred_idx] * 100
    
    return prediction, confidence

def print_result(text, label, confidence):
    """Print classification results in a clean, colored format."""
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}Message Preview:{Colors.END}")
    preview = text.strip().replace('\n', ' ')
    if len(preview) > 100:
        preview = preview[:97] + "..."
    print(f"  \"{preview}\"")
    
    color = Colors.RED if label == "Spam" else Colors.GREEN
    label_text = f"{color}{Colors.BOLD}{label.upper()}{Colors.END}"
    
    print(f"{Colors.BOLD}Classification Result:{Colors.END} {label_text}")
    print(f"{Colors.BOLD}Confidence Score:     {Colors.END}{Colors.CYAN}{confidence:.2f}%{Colors.END}")
    print("=" * 60 + "\n")

def interactive_mode(model, vectorizer):
    """Run an interactive prompt for entering text classification queries."""
    print(f"{Colors.HEADER}{Colors.BOLD}=== Spam Email Classifier (Interactive Mode) ==={Colors.END}")
    print("Type your email content below. Press Enter on an empty line to classify.")
    print("Type 'exit' or 'quit' to close the program.\n")
    
    while True:
        try:
            print(f"{Colors.BLUE}{Colors.BOLD}Enter Email Text (Ctrl+C to exit):{Colors.END}")
            lines = []
            while True:
                line = input()
                if not line and not lines:
                    continue
                if line.strip().lower() in ['exit', 'quit'] and not lines:
                    print("Exiting interactive mode. Goodbye!")
                    return
                if not line:
                    break
                lines.append(line)
            
            message = "\n".join(lines).strip()
            if not message:
                continue
                
            prediction, confidence = classify_text(message, model, vectorizer)
            print_result(message, prediction, confidence)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"{Colors.RED}An error occurred: {e}{Colors.END}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Classify email/SMS messages as 'Spam' or 'Not Spam'."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-m', '--message', 
        type=str, 
        help="Direct message string to classify."
    )
    group.add_argument(
        '-f', '--file', 
        type=str, 
        help="Path to a text file containing the message to classify."
    )
    
    args = parser.parse_args()
    
    model, vectorizer = load_classifier()
    
    if args.message:
        prediction, confidence = classify_text(args.message, model, vectorizer)
        print_result(args.message, prediction, confidence)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"{Colors.RED}Error: File '{args.file}' not found.{Colors.END}")
            sys.exit(1)
        try:
            with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            prediction, confidence = classify_text(content, model, vectorizer)
            print_result(content, prediction, confidence)
        except Exception as e:
            print(f"{Colors.RED}Error reading file: {e}{Colors.END}")
            sys.exit(1)
    else:
        interactive_mode(model, vectorizer)

if __name__ == "__main__":
    main()
