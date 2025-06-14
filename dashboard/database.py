import sqlite3
from datetime import datetime
import json
from typing import Dict, List, Optional
import pandas as pd
import hashlib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'data.db')

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

def register_user(email: str, password: str) -> bool:
    """Register a new user."""
    try:
        logger.info(f"Attempting to register user: {email}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if user already exists
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone() is not None:
            logger.info(f"User {email} already exists")
            return False
        
        # Insert new user
        password_hash = hash_password(password)
        c.execute(
            'INSERT INTO users (email, password_hash) VALUES (?, ?)',
            (email, password_hash)
        )
        conn.commit()
        logger.info(f"User {email} registered successfully")
        return True
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return False
    finally:
        conn.close()

def verify_user(email: str, password: str) -> Optional[int]:
    """Verify user credentials and return user_id if valid."""
    try:
        logger.info(f"Attempting to verify user: {email}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        password_hash = hash_password(password)
        c.execute(
            'SELECT id FROM users WHERE email = ? AND password_hash = ?',
            (email, password_hash)
        )
        result = c.fetchone()
        
        if result:
            user_id = result[0]
            # Update last login
            c.execute(
                'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                (user_id,)
            )
            conn.commit()
            logger.info(f"User {email} verified successfully")
            return user_id
        
        logger.info(f"Invalid credentials for user: {email}")
        return None
    except Exception as e:
        logger.error(f"Error verifying user: {e}")
        return None
    finally:
        conn.close()

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

def save_user_feedback(user_id: int, history_id: int, feedback_data: Dict) -> bool:
    """Save user feedback."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO user_feedback (
                user_id, history_id, feedback_type, specific_issues, suggestions
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            history_id,
            feedback_data.get('feedback_type'),
            feedback_data.get('specific_issues'),
            feedback_data.get('suggestions')
        ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False
    finally:
        conn.close()

def save_analysis(user_id: Optional[int], listing: Dict, results: Dict) -> int:
    """Save analysis results to database."""
    try:
        logger.info("Saving analysis results...")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Enable foreign key support
        c.execute('PRAGMA foreign_keys = ON')
        
        # Verify user exists if user_id is provided
        if user_id is not None:
            c.execute('SELECT id FROM users WHERE id = ?', (user_id,))
            if not c.fetchone():
                raise ValueError(f"User with id {user_id} does not exist")
        
        # Insert into analysis_history
        c.execute('''
            INSERT INTO analysis_history (
                user_id,
                job_title,
                job_description,
                location,
                company_profile,
                prediction,
                confidence_score,
                risk_factors,
                verification_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            listing['title'],
            listing['description'],
            listing.get('location'),
            listing.get('company_profile'),
            'Scam' if results['probability'] > 0.5 else 'Legitimate',
            results['probability'],
            json.dumps(results.get('risk_factors', [])),
            results.get('verification_score', 0.0)
        ))
        
        conn.commit()
        analysis_id = c.lastrowid
        logger.info(f"Analysis saved successfully with ID: {analysis_id}")
        return analysis_id
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        raise
    finally:
        conn.close()

def save_feedback(user_id: int, analysis_id: int, feedback_type: str, specific_issues: Optional[List[str]] = None, suggestions: Optional[str] = None) -> bool:
    """Save user feedback for an analysis."""
    try:
        logger.info(f"Saving feedback for analysis {analysis_id} from user {user_id}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Enable foreign key support
        c.execute('PRAGMA foreign_keys = ON')
        
        # Verify user and analysis exist
        c.execute('SELECT id FROM users WHERE id = ?', (user_id,))
        if not c.fetchone():
            raise ValueError(f"User with id {user_id} does not exist")
            
        c.execute('SELECT id FROM analysis_history WHERE id = ?', (analysis_id,))
        if not c.fetchone():
            raise ValueError(f"Analysis with id {analysis_id} does not exist")
        
        # Check if feedback already exists
        c.execute('''
            SELECT id FROM user_feedback 
            WHERE user_id = ? AND analysis_id = ?
        ''', (user_id, analysis_id))
        
        existing_feedback = c.fetchone()
        if existing_feedback:
            # Update existing feedback
            c.execute('''
                UPDATE user_feedback 
                SET feedback_type = ?, specific_issues = ?, suggestions = ?
                WHERE user_id = ? AND analysis_id = ?
            ''', (
                feedback_type,
                json.dumps(specific_issues) if specific_issues else None,
                suggestions,
                user_id,
                analysis_id
            ))
        else:
            # Insert new feedback
            c.execute('''
                INSERT INTO user_feedback (
                    user_id,
                    analysis_id,
                    feedback_type,
                    specific_issues,
                    suggestions
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                analysis_id,
                feedback_type,
                json.dumps(specific_issues) if specific_issues else None,
                suggestions
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

def get_feedback_stats() -> Dict:
    """Get feedback statistics."""
    try:
        logger.info("Retrieving feedback statistics...")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get feedback type distribution
        c.execute('''
            SELECT 
                feedback_type,
                COUNT(*) as count
            FROM user_feedback
            GROUP BY feedback_type
        ''')
        feedback_types = dict(c.fetchall())
        
        # Get total feedback count
        c.execute('SELECT COUNT(*) FROM user_feedback')
        total_feedback = c.fetchone()[0]
        
        # Get recent feedback with job details
        c.execute('''
            SELECT 
                h.job_title,
                f.feedback_type,
                f.specific_issues,
                f.suggestions,
                f.created_at
            FROM user_feedback f
            JOIN analysis_history h ON f.analysis_id = h.id
            ORDER BY f.created_at DESC
            LIMIT 5
        ''')
        
        recent_feedback = []
        for row in c.fetchall():
            feedback = {
                'job_title': row[0],
                'feedback_type': row[1],
                'specific_issues': json.loads(row[2]) if row[2] else [],
                'suggestions': row[3],
                'created_at': row[4]
            }
            recent_feedback.append(feedback)
        
        logger.info("Feedback statistics retrieved successfully")
        return {
            'feedback_types': feedback_types,
            'total_feedback': total_feedback,
            'recent_feedback': recent_feedback
        }
    except Exception as e:
        logger.error(f"Error retrieving feedback statistics: {e}")
        raise
    finally:
        conn.close()

def get_analysis_stats() -> Dict:
    """Get summary statistics of analyses."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get total analyses
    c.execute('SELECT COUNT(*) FROM analysis_history')
    total_analyses = c.fetchone()[0]
    
    # Get scam count
    c.execute('SELECT COUNT(*) FROM analysis_history WHERE prediction = "Scam"')
    scam_count = c.fetchone()[0]
    
    # Get average confidence
    c.execute('SELECT AVG(confidence_score) FROM analysis_history')
    avg_confidence = c.fetchone()[0] or 0
    
    # Get feedback distribution
    c.execute('''
        SELECT feedback_type, COUNT(*) 
        FROM user_feedback 
        GROUP BY feedback_type
    ''')
    feedback_dist = dict(c.fetchall())
    
    conn.close()
    
    return {
        'total_analyses': total_analyses,
        'scam_count': scam_count,
        'avg_confidence': avg_confidence,
        'feedback_dist': feedback_dist
    }

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