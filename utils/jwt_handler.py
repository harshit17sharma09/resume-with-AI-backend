from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException
import os
from dotenv import load_dotenv
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get secret key from environment variable
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create a JWT token."""
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Created access token for user: {data.get('sub', 'unknown')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating access token")

def decode_access_token(token: str, verify_expiration: bool = True):
    """Decode and validate JWT token."""
    try:
        logger.debug(f"Attempting to decode token: {token[:10]}...")  # Log first 10 chars
        
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            options={"verify_exp": verify_expiration}  # Configurable expiration verification
        )
        
        logger.debug(f"Token decoded successfully. Payload: {payload}")
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        if verify_expiration:
            raise HTTPException(
                status_code=401,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
        else:
            # If we're not verifying expiration, try to decode again without verification
            try:
                return jwt.decode(
                    token,
                    SECRET_KEY,
                    algorithms=[ALGORITHM],
                    options={"verify_exp": False}
                )
            except Exception as e:
                logger.error(f"Error decoding expired token: {str(e)}")
                raise HTTPException(status_code=401, detail="Invalid token format")
    except jwt.JWTError as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )