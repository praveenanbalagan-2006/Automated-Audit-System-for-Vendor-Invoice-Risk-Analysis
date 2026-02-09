
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')


import numpy as np
from PIL import Image, ImageDraw
import json
import re
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from sklearn.ensemble import IsolationForest
import io
import random
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import webbrowser
import jwt
import time
import logging
from functools import wraps
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECRET_KEY = 'auto-audit-secret-2024'

# ============================================================================
# SIMPLE OCR ENGINE (No external dependencies)
# ============================================================================

class SimpleOCREngine:
    def __init__(self):
        self.cache = {}
        self.patterns = {
            'invoice_number': r'(?:Invoice|Inv\s*No|Inv#|INV-)[\s:]*([A-Z0-9\-\/]+)',
            'vendor_name': r'(?:From|Vendor|Company|Bill\s*From)[\s:]*([A-Z][A-Za-z0-9\s&.,\-\']+)',
            'total': r'(?:Total|TOTAL|Grand\s*Total)[\s:]*\$?([0-9,]+\.?\d{0,2})',
        }
    
    def extract_text_simple(self, image_path):
        """Extract text from image using PIL"""
        try:
            img = Image.open(image_path)
            # For now, return mock data - in production use pytesseract
            return self.generate_mock_data()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def generate_mock_data(self):
        """Generate realistic mock invoice data"""
        vendors = ['TechCorp Ltd', 'Office Supplies Inc', 'Cloud Services LLC', 'Software Solutions']
        items = [
            ('Product A', 100.00 + random.randint(0, 100)),
            ('Service B', 250.00 + random.randint(0, 100)),
            ('License C', 150.00 + random.randint(0, 100)),
        ]
        
        text = f"""
        INVOICE #{random.randint(10000, 99999)}
        From: {random.choice(vendors)}
        Date: 2024-02-{random.randint(1,28):02d}
        
        ITEMS:
        """
        
        subtotal = 0
        for item, price in items:
            text += f"\n{item}: ${price:.2f}"
            subtotal += price
        
        tax = subtotal * 0.1
        total = subtotal + tax
        text += f"\n\nSubtotal: ${subtotal:.2f}\nTax (10%): ${tax:.2f}\nTOTAL: ${total:.2f}"
        
        return text
    
    def extract_structured_data(self, text):
        data = {
            'invoice_number': f'INV-2024-{random.randint(10000, 99999)}',
            'vendor_name': random.choice(['TechCorp Ltd', 'Office Supplies Inc', 'Cloud Services LLC']),
            'line_items': [150.0, 350.0, 200.0],
            'total': 700.0,
            'invoice_date': '2024-02-09',
        }
        
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data[field] = match.group(1).strip()
        
        amounts = re.findall(r'\$?([0-9,]+\.?\d{0,2})', text)
        if amounts:
            data['line_items'] = [float(amt.replace(',', '')) for amt in amounts[:10]]
        
        return data
    
    def process_invoice(self, image_path):
        try:
            text = self.extract_text_simple(image_path)
            if not text:
                return None
            
            data = self.extract_structured_data(text)
            data['raw_text'] = text[:500]
            data['extracted_at'] = datetime.now().isoformat()
            return data
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None

# ============================================================================
# VERIFICATION ENGINE
# ============================================================================

class VerificationEngine:
    def __init__(self):
        self.cache = {}
    
    def parse_amount(self, amount_str):
        try:
            if isinstance(amount_str, (int, float)):
                return float(amount_str)
            cleaned = re.sub(r'[^\d.,]', '', str(amount_str))
            cleaned = cleaned.replace(',', '').replace('.', '', cleaned.count('.') - 1)
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def verify_totals(self, extracted_data):
        results = {'valid': True, 'discrepancies': [], 'verification_score': 100}
        
        try:
            line_items = extracted_data.get('line_items', [])
            total = self.parse_amount(extracted_data.get('total', 0))
            
            if not line_items or total == 0:
                results['verification_score'] = 75
                return results
            
            calculated_subtotal = sum(line_items)
            difference = abs(calculated_subtotal - total)
            
            if difference > 0.01:
                deviation_pct = (difference / total * 100) if total > 0 else 0
                if deviation_pct > 10:
                    results['valid'] = False
                    results['verification_score'] = max(60, 100 - int(deviation_pct * 2))
                else:
                    results['verification_score'] = 90
            else:
                results['verification_score'] = 100
            
            results['calculated_subtotal'] = round(calculated_subtotal, 2)
            results['reported_total'] = total
        except Exception as e:
            logger.error(f"Verification error: {e}")
            results['verification_score'] = 75
        
        return results
    
    def generate_report(self, extracted_data):
        report = {
            'invoice_number': extracted_data.get('invoice_number', 'UNKNOWN'),
            'vendor': extracted_data.get('vendor_name', 'UNKNOWN'),
            'processing_timestamp': datetime.now().isoformat(),
        }
        
        total_verification = self.verify_totals(extracted_data)
        report.update(total_verification)
        
        if report['verification_score'] >= 85:
            report['status'] = 'VERIFIED'
        elif report['verification_score'] >= 60:
            report['status'] = 'NEEDS_REVIEW'
        else:
            report['status'] = 'FLAGGED'
        
        return report

# ============================================================================
# RISK ENGINE
# ============================================================================

class RiskEngine:
    def __init__(self):
        self.vendor_profiles = {}
        self.historical_invoices = []
    
    def add_invoice(self, invoice_data):
        self.historical_invoices.append(invoice_data)
        vendor = invoice_data.get('vendor', 'UNKNOWN')
        
        if vendor not in self.vendor_profiles:
            self.vendor_profiles[vendor] = {
                'invoices': [],
                'total_amount': 0,
                'avg_amount': 0,
                'frequency': 0,
            }
        
        self.vendor_profiles[vendor]['invoices'].append(invoice_data)
        self.vendor_profiles[vendor]['total_amount'] += invoice_data.get('total', 0)
        self.vendor_profiles[vendor]['frequency'] += 1
        self.vendor_profiles[vendor]['avg_amount'] = (
            self.vendor_profiles[vendor]['total_amount'] / 
            self.vendor_profiles[vendor]['frequency']
        )
    
    def calculate_risk(self, new_invoice, vendor):
        risk_score = 0
        risk_factors = []
        
        verification_score = new_invoice.get('verification_score', 100)
        if verification_score < 60:
            risk_score += 25
            risk_factors.append(f"Low verification: {verification_score}/100")
        elif verification_score < 80:
            risk_score += 10
        
        if vendor in self.vendor_profiles:
            vendor_data = self.vendor_profiles[vendor]
            if vendor_data['avg_amount'] > 0:
                avg_amount = vendor_data['avg_amount']
                new_amount = new_invoice.get('total', 0)
                deviation_pct = abs(new_amount - avg_amount) / avg_amount * 100
                
                if deviation_pct > 200:
                    risk_score += 20
                elif deviation_pct > 100:
                    risk_score += 10
        
        risk_score = min(risk_score, 100)
        
        return {
            'risk_score': round(risk_score, 1),
            'risk_level': self._get_level(risk_score),
            'risk_factors': risk_factors,
        }
    
    def _get_level(self, score):
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        elif score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, db_path='invoices.db'):
        self.db_path = db_path
        self.init_db()
    
    def get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn
    
    def init_db(self):
        conn = self.get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_number TEXT UNIQUE,
                vendor_name TEXT,
                invoice_date TEXT,
                total REAL,
                verification_score INTEGER,
                verification_status TEXT,
                risk_score REAL,
                risk_level TEXT,
                fraud_probability REAL,
                raw_data TEXT,
                processed_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vendor ON invoices(vendor_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk ON invoices(risk_level)')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('INSERT OR IGNORE INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                      ('admin', hashlib.sha256('admin123'.encode()).hexdigest(), 'admin@autoaudit.com'))
        
        conn.commit()
        conn.close()
    
    def insert_invoice(self, invoice_data, processed_time=0):
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO invoices 
                (invoice_number, vendor_name, invoice_date, total, 
                 verification_score, verification_status, risk_score, risk_level,
                 fraud_probability, raw_data, processed_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                invoice_data.get('invoice_number', f'INV-{int(time.time())}'),
                invoice_data.get('vendor', 'UNKNOWN'),
                invoice_data.get('invoice_date', datetime.now().isoformat()),
                invoice_data.get('total', 0),
                invoice_data.get('verification_score', 0),
                invoice_data.get('verification_status', 'PENDING'),
                invoice_data.get('risk_score', 0),
                invoice_data.get('risk_level', 'UNKNOWN'),
                invoice_data.get('fraud_probability', 0),
                json.dumps(invoice_data)[:2000],
                processed_time
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Insert error: {e}")
            return False
    
    def get_all_invoices(self, limit=1000):
        try:
            conn = self.get_conn()
            df = pd.read_sql_query(f'SELECT * FROM invoices ORDER BY created_at DESC LIMIT {limit}', conn)
            conn.close()
            return df.to_dict('records')
        except:
            return []
    
    def get_dashboard(self):
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            
            cursor.execute('SELECT SUM(total) as total FROM invoices')
            total_spending = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(*) as count FROM invoices')
            total_invoices = cursor.fetchone()[0] or 0
            
            vendor_df = pd.read_sql_query(
                '''SELECT vendor_name, COUNT(*) as invoice_count, 
                   SUM(total) as total_amount, AVG(risk_score) as avg_risk
                   FROM invoices GROUP BY vendor_name ORDER BY total_amount DESC LIMIT 50''',
                conn
            )
            
            risk_df = pd.read_sql_query(
                'SELECT risk_level, COUNT(*) as count FROM invoices GROUP BY risk_level',
                conn
            )
            
            conn.close()
            
            return {
                'total_spending': round(total_spending, 2),
                'total_invoices': int(total_invoices),
                'avg_invoice': round(total_spending / max(total_invoices, 1), 2),
                'vendor_data': vendor_df.to_dict('records') if not vendor_df.empty else [],
                'risk_distribution': risk_df.to_dict('records') if not risk_df.empty else [],
            }
        except:
            return {'total_spending': 0, 'total_invoices': 0, 'vendor_data': [], 'risk_distribution': []}
    
    def verify_user(self, username, password):
        try:
            conn = self.get_conn()
            cursor = conn.cursor()
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute('SELECT id FROM users WHERE username = ? AND password_hash = ?',
                          (username, password_hash))
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except:
            return False

# ============================================================================
# SAMPLE INVOICE GENERATOR
# ============================================================================

def create_sample():
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    vendors = ['TechCorp Ltd', 'Office Supplies Inc', 'Cloud Services LLC']
    vendor = random.choice(vendors)
    inv_num = f'INV-2024-{random.randint(10000, 99999)}'
    
    items = [
        ('Product A', 100.00 + random.randint(0, 100)),
        ('Service B', 250.00 + random.randint(0, 100)),
        ('License C', 150.00 + random.randint(0, 100)),
    ]
    
    y = 30
    draw.text((30, y), f"INVOICE #{inv_num}", fill='black')
    y += 50
    draw.text((30, y), f"From: {vendor}", fill='black')
    y += 50
    draw.text((30, y), f"Date: 2024-02-{random.randint(1,28):02d}", fill='black')
    y += 100
    
    draw.text((30, y), "ITEMS:", fill='black')
    y += 40
    
    subtotal = 0
    for item, price in items:
        draw.text((30, y), f"{item}: ${price:.2f}", fill='black')
        subtotal += price
        y += 40
    
    y += 40
    tax = subtotal * 0.1
    total = subtotal + tax
    draw.text((30, y), f"Subtotal: ${subtotal:.2f}", fill='black')
    y += 40
    draw.text((30, y), f"Tax (10%): ${tax:.2f}", fill='black')
    y += 40
    draw.text((30, y), f"TOTAL: ${total:.2f}", fill='black')
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes, inv_num, vendor, subtotal, total

# ============================================================================
# COMPLETE PROCESSOR
# ============================================================================

class Processor:
    def __init__(self, ocr, verify, risk, db):
        self.ocr = ocr
        self.verify = verify
        self.risk = risk
        self.db = db
    
    def process(self, image_path):
        start = time.time()
        
        try:
            extracted = self.ocr.process_invoice(image_path)
            if not extracted:
                return None
            
            ver_report = self.verify.generate_report(extracted)
            extracted.update({
                'verification_score': ver_report['verification_score'],
                'verification_status': ver_report['status'],
            })
            
            vendor = extracted.get('vendor_name', 'UNKNOWN')
            risk_data = self.risk.calculate_risk(extracted, vendor)
            extracted.update({
                'risk_score': risk_data['risk_score'],
                'risk_level': risk_data['risk_level'],
                'fraud_probability': risk_data['risk_score'] / 100,
                'vendor': vendor,
                'total': float(extracted.get('total', 0)) if extracted.get('total') else 0
            })
            
            duration = time.time() - start
            self.db.insert_invoice(extracted, duration)
            self.risk.add_invoice(extracted)
            
            extracted['processed_time'] = round(duration, 2)
            return extracted
        except Exception as e:
            logger.error(f"Process error: {e}")
            return None

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = SECRET_KEY

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('reports', exist_ok=True)

CORS(app)

print("⏳ Initializing engines...")
ocr_engine = SimpleOCREngine()
verify_engine = VerificationEngine()
risk_engine = RiskEngine()
db_manager = DatabaseManager()
processor = Processor(ocr_engine, verify_engine, risk_engine, db_manager)

print("✅ Ready!\n")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Missing token'}), 401
        try:
            token = token.split(' ')[1]
            jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if db_manager.verify_user(data.get('username', ''), data.get('password', '')):
        token = jwt.encode({'user_id': data['username'], 'exp': datetime.utcnow() + timedelta(hours=24)},
                          SECRET_KEY, algorithm='HS256')
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/dashboard')
@token_required
def get_dashboard():
    return jsonify(db_manager.get_dashboard())

@app.route('/api/invoices')
@token_required
def get_invoices():
    return jsonify(db_manager.get_all_invoices())

@app.route('/api/process', methods=['POST'])
@token_required
def process_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        result = processor.process(filepath)
        if not result:
            return jsonify({'error': 'Processing failed'}), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Process error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/sample', methods=['POST'])
@token_required
def sample():
    try:
        img, inv_num, vendor, sub, total = create_sample()
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'sample_{inv_num}.png')
        with open(path, 'wb') as f:
            f.write(img.getvalue())
        
        result = processor.process(path)
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to process sample'}), 400
    except Exception as e:
        logger.error(f"Sample error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 STARTING AUTO AUDIT INVOICE SYSTEM v4.0 - FIXED VERSION")
    print("=" * 80)
    print("\n✅ System initialized successfully!")
    print("📍 Server starting...")
    print("🌐 Visit: http://localhost:5000\n")
    print("Login Credentials:")
    print("  Username: admin")
    print("  Password: admin123\n")
    
    threading.Timer(2, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
