import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to Firebase Admin SDK JSON file
cred = credentials.Certificate(os.getenv("FIREBASE_ADMIN_CREDENTIALS"))
firebase_admin.initialize_app(cred)

# Firestore instance
db = firestore.client()
