from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

MODEL_NAME = "bert-base-uncased"
tokenizer = None
model = None

def load_dataset():
    """Load fake and real news from CSV files"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fake_path = os.path.join(script_dir, 'News _dataset', 'Fake.csv')
        true_path = os.path.join(script_dir, 'News _dataset', 'True.csv')
        
        print(f"Loading fake news from: {fake_path}")
        print(f"Loading true news from: {true_path}")
        
        # Load datasets with error handling for encoding issues
        fake_df = pd.read_csv(fake_path, engine='python', encoding='utf-8', on_bad_lines='skip')
        true_df = pd.read_csv(true_path, engine='python', encoding='utf-8', on_bad_lines='skip')
        
        print(f"  ✓ Loaded {len(fake_df)} fake news articles")
        print(f"  ✓ Loaded {len(true_df)} true news articles")
        
        # Combine title and text for better context
        fake_df['combined'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
        true_df['combined'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
        
        # Create labels (1=Fake, 0=Real)
        fake_df['label'] = 1
        true_df['label'] = 0
        
        # Combine datasets
        combined_df = pd.concat([fake_df[['combined', 'label']], true_df[['combined', 'label']]], ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"  ✓ Total dataset size: {len(combined_df)} articles")
        
        return combined_df['combined'].tolist(), combined_df['label'].tolist()
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def fine_tune_model(model_to_train, train_texts, train_labels, val_texts, val_labels):
    """Fine-tune BERT on fake and real news dataset"""
    
    print(f"\nTraining on {len(train_texts)} samples, validating on {len(val_texts)} samples")
    
    # Training setup
    model_to_train.train()
    optimizer = AdamW(model_to_train.parameters(), lr=2e-5)
    
    # Training parameters
    batch_size = 8  # Process 8 articles at a time
    num_epochs = 3  # 3 epochs for large dataset
    
    print(f"\nTraining BERT model on News Dataset...")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: 2e-5\n")
    
    for epoch in range(num_epochs):
        model_to_train.train()
        total_loss = 0
        correct = 0
        num_batches = 0
        
        # Training loop with batches
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            labels = torch.tensor(batch_labels)
            
            # Forward pass
            outputs = model_to_train(**inputs, labels=labels)
            loss = outputs.loss
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if num_batches % 100 == 0:
                print(f"  Epoch {epoch + 1}/{num_epochs} - Batch {num_batches}/{len(train_texts)//batch_size} - Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        avg_loss = total_loss / num_batches
        train_accuracy = (correct / len(train_texts)) * 100
        
        # Validation
        model_to_train.eval()
        val_correct = 0
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch_texts = val_texts[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                labels = torch.tensor(batch_labels)
                
                outputs = model_to_train(**inputs, labels=labels)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                val_correct += (preds == labels).sum().item()
                val_loss += outputs.loss.item()
                val_batches += 1
        
        val_accuracy = (val_correct / len(val_texts)) * 100
        val_avg_loss = val_loss / val_batches
        
        print(f"\n  📊 Epoch {epoch + 1}/{num_epochs} Results:")
        print(f"     Training   - Loss: {avg_loss:.4f} | Accuracy: {train_accuracy:.2f}%")
        print(f"     Validation - Loss: {val_avg_loss:.4f} | Accuracy: {val_accuracy:.2f}%\n")
    
    model_to_train.eval()
    print("✓ BERT fine-tuning complete! Model ready for predictions.")
    return model_to_train

def extract_features(text):
    """Extract linguistic features for display purposes"""
    features = {
        'sensationalWords': len(re.findall(
            r'BREAKING|SHOCKING|UNBELIEVABLE|MIRACLE|DISCOVER|NEVER|ALWAYS|UNEARTHED|REWRITES', 
            text, re.IGNORECASE
        )),
        'exclamationMarks': text.count('!'),
        'allCaps': len(re.findall(r'\b[A-Z]{4,}\b', text)),
        'fictionalTerms': len(re.findall(
            r'fictional|mythical|imaginary|made-up|fake|hoax', 
            text, re.IGNORECASE
        )),
        'unrealisticClaims': len(re.findall(
            r'teleportation|time.travel|immortal|anti.gravity|unlimited.energy', 
            text, re.IGNORECASE
        )),
        'scientificJargon': len(re.findall(
            r'chrono|quantum|nano|hyper|mega|ultra', 
            text, re.IGNORECASE
        )),
        'emojis': len(re.findall(r'[\U0001F300-\U0001F9FF]', text)),
        'length': len(text),
        'questionMarks': text.count('?'),
        'hasNumbers': bool(re.search(r'\d', text))
    }
    return features

def analyze_with_bert(text):
    """Analyze text using FINE-TUNED BERT model"""
    
    print(f"\n🤖 Analyzing with BERT ({len(text)} characters)...")
    
    if model is None or tokenizer is None:
        return {"error": "BERT model not available"}
    
    try:
        # Tokenize input for BERT
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Get BERT predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction (0 = Real, 1 = Fake)
        prediction_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction_idx].item() * 100
        
        # Get both probabilities
        fake_probability = probabilities[0][1].item() * 100
        real_probability = probabilities[0][0].item() * 100
        
        print(f"  🔍 BERT Analysis:")
        print(f"     - Fake probability: {fake_probability:.1f}%")
        print(f"     - Real probability: {real_probability:.1f}%")
        print(f"     - Final prediction: {'FAKE' if prediction_idx == 1 else 'REAL'}")
        
        # Extract features for display
        features = extract_features(text)
        
        # Calculate breakdown scores based on BERT confidence
        if prediction_idx == 1:  # Fake
            linguistic_score = max(20, 100 - fake_probability * 0.6)
            semantic_score = max(15, 100 - fake_probability * 0.7)
            contextual_score = max(10, 100 - fake_probability * 0.8)
        else:  # Real
            linguistic_score = min(90, 50 + real_probability * 0.4)
            semantic_score = min(95, 50 + real_probability * 0.5)
            contextual_score = min(85, 50 + real_probability * 0.35)
        
        # Determine sentiment based on BERT prediction
        if prediction_idx == 1:  # Fake
            sentiment = "Highly Sensational"
            credibility = "Low"
            emotional_tone = "Manipulative"
        else:  # Real
            sentiment = "Neutral/Factual"
            credibility = "High"
            emotional_tone = "Informative"
        
        return {
            'prediction': 'Likely Fake' if prediction_idx == 1 else 'Likely Real',
            'confidence': round(confidence, 1),
            'features': features,
            'analysis': {
                'sentiment': sentiment,
                'credibility': credibility,
                'emotionalTone': emotional_tone
            },
            'breakdown': {
                'linguistic': round(linguistic_score, 1),
                'semantic': round(semantic_score, 1),
                'contextual': round(contextual_score, 1)
            },
            'model_info': {
                'model_used': 'BERT (bert-base-uncased) Fine-tuned on Fake News Dataset',
                'fake_probability': round(fake_probability, 1),
                'real_probability': round(real_probability, 1),
                'model_type': 'Deep Learning - Transformer'
            }
        }
        
    except Exception as e:
        print(f"BERT analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Initialize model at startup
print("="*60)
print("🚀 Initializing BERT for Fake News Detection")
print("="*60)

try:
    print("Loading BERT tokenizer and base model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    print("  ✓ BERT model loaded successfully!\n")
    
    print("Loading News Dataset...")
    all_texts, all_labels = load_dataset()
    
    if all_texts is None or all_labels is None:
        raise Exception("Failed to load dataset")
    
    # Split dataset: 80% training, 20% validation
    # Using a smaller subset for faster training (you can increase this)
    subset_size = min(5000, len(all_texts))  # Use 5000 samples or all if less
    print(f"\nUsing {subset_size} samples for training (from {len(all_texts)} total)")
    print("Note: You can increase subset_size in the code for more comprehensive training\n")
    
    texts_subset = all_texts[:subset_size]
    labels_subset = all_labels[:subset_size]
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts_subset, labels_subset, test_size=0.2, random_state=42, stratify=labels_subset
    )
    
    print(f"Dataset split: {len(train_texts)} training, {len(val_texts)} validation")
    
    print("\nFine-tuning BERT on News Dataset...")
    print("This may take several minutes...\n")
    model = fine_tune_model(model, train_texts, train_labels, val_texts, val_labels)
    
    print("\n✓ BERT model loaded and fine-tuned successfully!")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    model = None
    tokenizer = None

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint for fake news analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze with BERT
        result = analyze_with_bert(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': MODEL_NAME,
        'model_type': 'BERT Fine-tuned',
        'version': '2.0.0'
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Fake News Detection API - BERT Fine-tuned',
        'model': 'BERT Transformer (Fine-tuned)',
        'endpoints': {
            '/api/analyze': 'POST - Analyze text for fake news',
            '/api/health': 'GET - Check API health'
        }
    }), 200

if __name__ == '__main__':
    print("🌐 Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)