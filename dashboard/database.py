import sqlite3
from datetime import datetime
import json
from typing import Dict, List, Optional
import pandas as pd
import hashlib
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up database path
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DB_DIR, 'job_scam.db')

# Ensure database directory exists
os.makedirs(DB_DIR, exist_ok=True)

def get_db():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize database with required tables."""
    conn = get_db()
    c = conn.cursor()
    
    try:
        # Enable foreign key support
        c.execute('PRAGMA foreign_keys = ON')
        
        # Create users table
        logger.info("Creating users table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_demo INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create analysis_history table
        logger.info("Creating analysis_history table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                job_title TEXT NOT NULL,
                job_description TEXT NOT NULL,
                location TEXT,
                company_profile TEXT,
                prediction TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                risk_factors TEXT,
                verification_score REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Create user_feedback table
        logger.info("Creating user_feedback table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                analysis_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                specific_issues TEXT,
                suggestions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history (id) ON DELETE CASCADE
            )
        ''')
        
        # Drop old feedback table if it exists
        c.execute("DROP TABLE IF EXISTS feedback")
        
        conn.commit()
        
        # Verify tables were created
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        logger.info(f"Created tables: {[table[0] for table in tables]}")
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def register_user(email, password):
    """Register a new user."""
    conn = get_db()
    c = conn.cursor()
    
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        c.execute(
            'INSERT INTO users (email, password_hash) VALUES (?, ?)',
            (email, password_hash)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email: str, password: str) -> Optional[int]:
    """Verify user credentials and return user_id if valid."""
    try:
        # Special case for demo account
        if email == "demo_email" and password == "demo_password":
            # Get or create demo user
            conn = get_db()
            c = conn.cursor()
            
            # Check if demo user exists
            c.execute('SELECT id FROM users WHERE email = ?', ('demo@example.com',))
            result = c.fetchone()
            
            if result:
                return result[0]
            else:
                # Create demo user if doesn't exist
                c.execute('''
                    INSERT INTO users (email, password_hash, is_demo)
                    VALUES (?, ?, ?)
                ''', ('demo@example.com', 'demo_hash', 1))
                conn.commit()
                return c.lastrowid
        
        # Regular user verification
        conn = get_db()
        c = conn.cursor()
        
        # Get user with matching email
        c.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
        result = c.fetchone()
        
        if not result:
            logger.warning(f"No user found with email {email}")
            return None
            
        user_id, stored_hash = result
        
        # Verify password
        if verify_password(password, stored_hash):
            return user_id
            
        logger.warning("Invalid password provided")
        return None
        
    except Exception as e:
        logger.error(f"Error verifying user: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify if provided password matches stored hash."""
    try:
        # For demo account
        if stored_hash == 'demo_hash' and password == 'demo_password':
            return True
            
        # For regular accounts
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == stored_hash
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

def save_analysis(user_id, description, prediction, confidence):
    """Save analysis result to history."""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO analysis_history 
        (user_id, description, prediction, confidence)
        VALUES (?, ?, ?, ?)
    ''', (user_id, description, prediction, confidence))
    
    conn.commit()
    conn.close()

def get_user_history(user_id):
    """Get analysis history for a user."""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM analysis_history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    ''', (user_id,))
    
    history = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return history

def save_feedback(user_id, accuracy_rating, usefulness_rating, false_prediction, comments):
    """Save user feedback."""
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO feedback 
        (user_id, accuracy_rating, usefulness_rating, false_prediction, comments)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, accuracy_rating, usefulness_rating, false_prediction, comments))
    
    conn.commit()
    conn.close()

def get_feedback_stats() -> Dict:
    """Get feedback statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        stats = {
            'total_feedback': 0,
            'feedback_types': {},
            'false_positives': 0.0,
            'false_negatives': 0.0,
            'recent_feedback': []
        }
        
        # Get total feedback count
        c.execute('SELECT COUNT(*) FROM user_feedback')
        stats['total_feedback'] = c.fetchone()[0]
        
        if stats['total_feedback'] > 0:
            # Calculate feedback type distribution
            c.execute('''
                SELECT feedback_type, COUNT(*) as count
                FROM user_feedback 
                GROUP BY feedback_type
            ''')
            stats['feedback_types'] = dict(c.fetchall())
            
            # Calculate error rates
            total = stats['total_feedback']
            c.execute('''
                SELECT 
                    COUNT(*) as total_errors,
                    SUM(CASE 
                        WHEN h.prediction = 'Scam' AND f.feedback_type = '❌ Incorrect' THEN 1
                        WHEN h.prediction = 'Scam' AND f.feedback_type = '⚠️ Partially Correct' THEN 0.5
                        ELSE 0 
                    END) as false_positives,
                    SUM(CASE 
                        WHEN h.prediction = 'Legitimate' AND f.feedback_type = '❌ Incorrect' THEN 1
                        WHEN h.prediction = 'Legitimate' AND f.feedback_type = '⚠️ Partially Correct' THEN 0.5
                        ELSE 0 
                    END) as false_negatives
                FROM user_feedback f
                JOIN analysis_history h ON f.analysis_id = h.id
            ''')
            
            error_stats = c.fetchone()
            if error_stats and total > 0:
                stats['false_positives'] = (error_stats[1] or 0) * 100.0 / total
                stats['false_negatives'] = (error_stats[2] or 0) * 100.0 / total
            
            # Get recent feedback with job titles
            c.execute('''
                SELECT 
                    f.feedback_type,
                    f.specific_issues,
                    f.suggestions,
                    f.created_at,
                    h.job_title
                FROM user_feedback f
                JOIN analysis_history h ON f.analysis_id = h.id
                ORDER BY f.created_at DESC
                LIMIT 5
            ''')
            
            stats['recent_feedback'] = [
                {
                    'feedback_type': row[0],
                    'specific_issues': row[1],
                    'suggestions': row[2],
                    'created_at': row[3],
                    'job_title': row[4]
                }
                for row in c.fetchall()
            ]
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving feedback statistics: {e}")
        raise
    finally:
        conn.close()

# Initialize database when module is imported
init_db()

def init_database():
    """Initialize the database with required tables."""
    try:
        logger.info(f"Initializing database at {DB_PATH}")
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Enable foreign key support
        c.execute('PRAGMA foreign_keys = ON')
        
        # Create users table first (no dependencies)
        logger.info("Creating users table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create analysis_history table (depends on users)
        logger.info("Creating analysis_history table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                job_title TEXT NOT NULL,
                job_description TEXT NOT NULL,
                location TEXT,
                company_profile TEXT,
                prediction TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                risk_factors TEXT,
                verification_score REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Create user_feedback table (depends on both users and analysis_history)
        logger.info("Creating user_feedback table...")
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                analysis_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                specific_issues TEXT,
                suggestions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (analysis_id) REFERENCES analysis_history (id) ON DELETE CASCADE
            )
        ''')
        
        # Verify tables were created
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        logger.info(f"Created tables: {[table[0] for table in tables]}")
        
        conn.commit()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def save_user_analysis(user_id: int, listing: Dict, results: Dict) -> int:
    """Save analysis results for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO user_history (
                user_id, job_title, job_description, location, company_profile,
                prediction, confidence_score, risk_factors, verification_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            listing.get('title'),
            listing.get('description'),
            listing.get('location'),
            listing.get('company_profile'),
            'Scam' if results['probability'] > 0.5 else 'Legitimate',
            results['probability'],
            ','.join(results.get('risk_factors', [])),
            results.get('verification_score', 0.0)
        ))
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()

def get_user_history(user_id: int) -> List[Dict]:
    """Get analysis history for a specific user."""
    try:
        logger.info(f"Retrieving history for user {user_id}")
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT 
                h.id,
                h.job_title,
                h.job_description,
                h.location,
                h.company_profile,
                h.prediction,
                h.confidence_score,
                h.risk_factors,
                h.verification_score,
                h.analysis_date,
                f.feedback_type,
                f.specific_issues,
                f.suggestions,
                f.created_at as feedback_created_at
            FROM analysis_history h
            LEFT JOIN user_feedback f ON h.id = f.analysis_id
            WHERE h.user_id = ?
            ORDER BY h.analysis_date DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id,))
        
        # Convert DataFrame to list of dictionaries
        history = []
        for _, row in df.iterrows():
            entry = row.to_dict()
            
            # Parse JSON fields
            try:
                entry['risk_factors'] = json.loads(entry['risk_factors']) if entry['risk_factors'] else []
            except:
                entry['risk_factors'] = []
            
            try:
                entry['specific_issues'] = json.loads(entry['specific_issues']) if entry['specific_issues'] else []
            except:
                entry['specific_issues'] = []
            
            history.append(entry)
        
        logger.info(f"Retrieved {len(history)} analysis records for user {user_id}")
        return history
    except Exception as e:
        logger.error(f"Error retrieving user history: {e}")
        raise
    finally:
        conn.close()

def save_user_feedback(user_id: int, analysis_id: int, feedback_data: Dict) -> bool:
    """Save user feedback."""
    try:
        logger.info(f"Saving feedback for analysis {analysis_id} from user {user_id}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Enable foreign key support
        c.execute('PRAGMA foreign_keys = ON')
        
        # Validate analysis_id exists
        c.execute('SELECT id FROM analysis_history WHERE id = ?', (analysis_id,))
        if not c.fetchone():
            logger.error(f"Analysis ID {analysis_id} not found")
            return False
            
        # Validate user_id exists
        c.execute('SELECT id FROM users WHERE id = ?', (user_id,))
        if not c.fetchone():
            logger.error(f"User ID {user_id} not found")
            return False
        
        # Check if feedback already exists
        c.execute('''
            SELECT id FROM user_feedback 
            WHERE user_id = ? AND analysis_id = ?
        ''', (user_id, analysis_id))
        
        existing_feedback = c.fetchone()
        
        # Prepare feedback data
        feedback_type = feedback_data.get('feedback_type')
        specific_issues = json.dumps(feedback_data.get('specific_issues', [])) if feedback_data.get('specific_issues') else None
        suggestions = feedback_data.get('suggestions')
        
        if existing_feedback:
            # Update existing feedback
            logger.info(f"Updating existing feedback for analysis {analysis_id}")
            c.execute('''
                UPDATE user_feedback 
                SET feedback_type = ?,
                    specific_issues = ?,
                    suggestions = ?,
                    created_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND analysis_id = ?
            ''', (feedback_type, specific_issues, suggestions, user_id, analysis_id))
        else:
            # Insert new feedback
            logger.info(f"Inserting new feedback for analysis {analysis_id}")
            c.execute('''
                INSERT INTO user_feedback (
                    user_id,
                    analysis_id,
                    feedback_type,
                    specific_issues,
                    suggestions,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, analysis_id, feedback_type, specific_issues, suggestions))
        
        conn.commit()
        logger.info(f"Feedback saved successfully for analysis {analysis_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database error while saving feedback: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False
    finally:
        conn.close()

def save_analysis(user_id: Optional[int], listing: Dict, results: Dict) -> int:
    """Save analysis results."""
    try:
        logger.info("Saving analysis results")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Extract prediction and risk factors
        prediction = 'Scam' if results['probability'] > 0.5 else 'Legitimate'
        risk_factors = json.dumps(results.get('risk_factors', []))
        
        c.execute('''
            INSERT INTO analysis_history (
                user_id, job_title, job_description, location, company_profile,
                prediction, confidence_score, risk_factors, verification_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            listing.get('title'),
            listing.get('description'),
            listing.get('location'),
            listing.get('company_profile'),
            prediction,
            results['probability'],
            risk_factors,
            results.get('verification_score', 0.0)
        ))
        
        analysis_id = c.lastrowid
        conn.commit()
        logger.info(f"Analysis saved with ID: {analysis_id}")
        return analysis_id
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        raise
    finally:
        conn.close()

def save_feedback(feedback_data: Dict) -> bool:
    """Save user feedback."""
    try:
        logger.info("Saving user feedback")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get the latest analysis for this job title
        c.execute('''
            SELECT id FROM analysis_history 
            WHERE job_title = ? 
            ORDER BY analysis_date DESC 
            LIMIT 1
        ''', (feedback_data['job_title'],))
        
        result = c.fetchone()
        if not result:
            logger.error("No matching analysis found for feedback")
            return False
            
        analysis_id = result[0]
        
        c.execute('''
            INSERT INTO user_feedback (
                analysis_id,
                feedback_type,
                specific_issues,
                suggestions,
                created_at
            ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            analysis_id,
            feedback_data['user_feedback'],
            feedback_data.get('specific_issues', ''),
            feedback_data.get('suggested_improvements', '')
        ))
        
        conn.commit()
        logger.info("Feedback saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False
    finally:
        conn.close()

def get_analysis_history() -> List[Dict]:
    """Get all analysis history."""
    try:
        logger.info("Retrieving all analysis history...")
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT 
                h.id,
                h.job_title,
                h.job_description,
                h.location,
                h.company_profile,
                h.prediction,
                h.confidence_score,
                h.risk_factors,
                h.verification_score,
                h.analysis_date,
                f.feedback_type,
                f.specific_issues,
                f.suggestions,
                f.created_at as feedback_created_at
            FROM analysis_history h
            LEFT JOIN user_feedback f ON h.id = f.analysis_id
            ORDER BY h.analysis_date DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        
        # Convert DataFrame to list of dictionaries
        history = []
        for _, row in df.iterrows():
            entry = row.to_dict()
            
            # Parse JSON fields
            try:
                entry['risk_factors'] = json.loads(entry['risk_factors']) if entry['risk_factors'] else []
            except:
                entry['risk_factors'] = []
            
            try:
                entry['specific_issues'] = json.loads(entry['specific_issues']) if entry['specific_issues'] else []
            except:
                entry['specific_issues'] = []
            
            history.append(entry)
        
        logger.info(f"Retrieved {len(history)} total analysis records")
        return history
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {e}")
        raise
    finally:
        conn.close()

def get_analysis_stats() -> Dict:
    """Get analysis statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        stats = {
            'total_analyzed': 0,
            'scam_count': 0,
            'legitimate_count': 0,
            'avg_confidence': 0.0
        }
        
        c.execute('SELECT COUNT(*) FROM analysis_history')
        stats['total_analyzed'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM analysis_history WHERE prediction = "Scam"')
        stats['scam_count'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM analysis_history WHERE prediction = "Legitimate"')
        stats['legitimate_count'] = c.fetchone()[0]
        
        c.execute('SELECT AVG(confidence_score) FROM analysis_history')
        stats['avg_confidence'] = c.fetchone()[0] or 0.0
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving analysis statistics: {e}")
        raise
    finally:
        conn.close()

def search_history(query: str, field: str) -> List[Dict]:
    """Search analysis history based on query and field."""
    conn = sqlite3.connect(DB_PATH)
    
    # Construct query with parameterized SQL
    sql_query = f'''
        SELECT 
            h.*,
            f.feedback_type,
            f.specific_issues,
            f.suggestions,
            f.feedback_date
        FROM analysis_history h
        LEFT JOIN user_feedback f ON h.id = f.analysis_id
        WHERE h.{field} LIKE ?
        ORDER BY h.analysis_date DESC
    '''
    
    df = pd.read_sql_query(sql_query, conn, params=[f'%{query}%'])
    conn.close()
    
    # Convert DataFrame to list of dictionaries
    history = []
    for _, row in df.iterrows():
        entry = row.to_dict()
        
        # Parse JSON fields
        try:
            entry['risk_factors'] = json.loads(entry['risk_factors']) if entry['risk_factors'] else []
        except:
            entry['risk_factors'] = []
            
        try:
            entry['specific_issues'] = json.loads(entry['specific_issues']) if entry['specific_issues'] else []
        except:
            entry['specific_issues'] = []
        
        history.append(entry)
    
    return history

def get_model_performance_metrics() -> Dict:
    """Calculate model performance metrics including F1 score from feedback data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        metrics = {
            'total_predictions': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'weighted_f1_score': 0.0,
            'balanced_accuracy': 0.0,
            'specificity': 0.0,
            'negative_predictive_value': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'class_distribution': {
                'scam': 0,
                'legitimate': 0
            }
        }
        
        # Get feedback counts
        c.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE 
                    WHEN h.prediction = 'Scam' AND f.feedback_type = '✅ Correct' THEN 1 
                    ELSE 0 
                END) as true_positives,
                SUM(CASE 
                    WHEN h.prediction = 'Scam' AND f.feedback_type != '✅ Correct' THEN 1 
                    ELSE 0 
                END) as false_positives,
                SUM(CASE 
                    WHEN h.prediction = 'Legitimate' AND f.feedback_type != '✅ Correct' THEN 1 
                    ELSE 0 
                END) as false_negatives,
                SUM(CASE 
                    WHEN h.prediction = 'Legitimate' AND f.feedback_type = '✅ Correct' THEN 1 
                    ELSE 0 
                END) as true_negatives,
                SUM(CASE WHEN h.prediction = 'Scam' THEN 1 ELSE 0 END) as total_scam,
                SUM(CASE WHEN h.prediction = 'Legitimate' THEN 1 ELSE 0 END) as total_legitimate
            FROM user_feedback f
            JOIN analysis_history h ON f.analysis_id = h.id
        ''')
        
        result = c.fetchone()
        if result and result[0] > 0:
            metrics['total_predictions'] = result[0]
            metrics['true_positives'] = result[1] or 0
            metrics['false_positives'] = result[2] or 0
            metrics['false_negatives'] = result[3] or 0
            metrics['true_negatives'] = result[4] or 0
            metrics['class_distribution']['scam'] = result[5] or 0
            metrics['class_distribution']['legitimate'] = result[6] or 0
            
            # Calculate basic metrics
            if (metrics['true_positives'] + metrics['false_positives']) > 0:
                metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            
            if (metrics['true_positives'] + metrics['false_negatives']) > 0:
                metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            
            # Calculate accuracy
            metrics['accuracy'] = (metrics['true_positives'] + metrics['true_negatives']) / metrics['total_predictions']
            
            # Calculate additional metrics
            if (metrics['true_negatives'] + metrics['false_positives']) > 0:
                metrics['specificity'] = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives'])
            
            if (metrics['true_negatives'] + metrics['false_negatives']) > 0:
                metrics['negative_predictive_value'] = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_negatives'])
            
            # Calculate error rates
            if (metrics['true_negatives'] + metrics['false_positives']) > 0:
                metrics['false_positive_rate'] = metrics['false_positives'] / (metrics['true_negatives'] + metrics['false_positives'])
            
            if (metrics['true_positives'] + metrics['false_negatives']) > 0:
                metrics['false_negative_rate'] = metrics['false_negatives'] / (metrics['true_positives'] + metrics['false_negatives'])
            
            # Calculate balanced metrics
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
            
            # Calculate weighted F1 score
            total_samples = metrics['class_distribution']['scam'] + metrics['class_distribution']['legitimate']
            if total_samples > 0:
                scam_weight = metrics['class_distribution']['scam'] / total_samples
                legitimate_weight = metrics['class_distribution']['legitimate'] / total_samples
                
                if metrics['precision'] + metrics['recall'] > 0:
                    scam_f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                    legitimate_f1 = 2 * (metrics['negative_predictive_value'] * metrics['specificity']) / (metrics['negative_predictive_value'] + metrics['specificity'])
                    metrics['weighted_f1_score'] = (scam_weight * scam_f1 + legitimate_weight * legitimate_f1)
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating model performance metrics: {e}")
        raise
    finally:
        conn.close() 