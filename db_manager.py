import sqlite3
from datetime import datetime

class DBManager:
    def __init__(self):
        self.db_file = "human_logs.db"
        self.connection = None
        self.connected = False
        
        self.connect()
        self.create_table()

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
            self.connected = True
            print(f"[DB] Connected to SQLite database: {self.db_file}")
                
        except sqlite3.Error as e:
            print(f"[DB] Error connecting to SQLite: {e}")
            self.connected = False

    def create_table(self):
        if not self.connected: return
        
        try:
            query = """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                timestamp DATETIME,
                details TEXT,
                avg_accuracy REAL
            )
            """
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"[DB] Error creating table: {e}")

    def log_event(self, event_type, details="", avg_accuracy=0.0):
        if not self.connected: 
            return

        try:
            query = "INSERT INTO logs (event_type, timestamp, details, avg_accuracy) VALUES (?, ?, ?, ?)"
            timestamp = datetime.now()
            val = (event_type, timestamp, details, avg_accuracy)
            
            cursor = self.connection.cursor()
            cursor.execute(query, val)
            self.connection.commit()
            print(f"[DB] Logged: {event_type}")
            
        except sqlite3.Error as e:
            print(f"[DB] Error logging event: {e}")

    def log_system_start(self):
        self.log_event("SYSTEM_START", "Application started")

    def log_no_face(self, avg_accuracy=0.0):
        self.log_event("NO_FACE_DETECTED", "No face detected for 15s+", avg_accuracy)
        
    def log_face_recovered(self, avg_accuracy=0.0):
        self.log_event("FACE_RECOVERED", "Face detected again (5s avg)", avg_accuracy)

    def log_phone_detected(self, avg_accuracy):
        self.log_event("PHONE_DETECTED", "Phone detected for 15s+", avg_accuracy)
        
    def log_phone_removed(self, avg_accuracy):
        self.log_event("PHONE_REMOVED", "Phone removed (5s avg)", avg_accuracy)

    def log_eyes_closed(self, avg_accuracy):
        self.log_event("EYES_CLOSED", "Eyes closed for 15s+", avg_accuracy)

    def log_eyes_opened(self, avg_accuracy):
        self.log_event("EYES_OPENED", "Eyes opened again (5s avg)", avg_accuracy)

    def log_too_far(self, avg_accuracy):
        self.log_event("TOO_FAR", "Person too far for 15s+", avg_accuracy)

    def log_back_in_range(self, avg_accuracy):
        self.log_event("BACK_IN_RANGE", "Person back in range (5s avg)", avg_accuracy)
        
    def fetch_logs(self, limit=50):
        if not self.connected: return []
        try:
            query = f"SELECT event_type, timestamp, details, avg_accuracy FROM logs ORDER BY timestamp DESC LIMIT {limit}"
            cursor = self.connection.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            print(f"[DB] Error fetching logs: {e}")
            return []
            
    def clear_logs(self):
        if not self.connected: return False
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM logs")
            self.connection.commit()
            print("[DB] All logs cleared.")
            return True
        except Exception as e:
            print(f"[DB] Error clearing logs: {e}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()
