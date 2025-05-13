from datetime import datetime, timedelta
import logging
import hmac
import secrets
from ipaddress import ip_address, ip_network
import time
from typing import Optional, Dict, List, Set

import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .config import global_args

# Configure logger
logger = logging.getLogger(__name__)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# API key security scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate limiting tracker
rate_limiter: Dict[str, Dict] = {}


class TokenPayload(BaseModel):
    sub: str  # Username
    exp: datetime  # Expiration time
    role: str = "user"  # User role, default is regular user
    metadata: dict = {}  # Additional metadata


class AuthHandler:
    def __init__(self):
        self.secret = global_args.token_secret
        self.algorithm = global_args.jwt_algorithm
        self.expire_hours = global_args.token_expire_hours
        self.guest_expire_hours = global_args.guest_token_expire_hours
        self.accounts = {}
        auth_accounts = global_args.auth_accounts
        if auth_accounts:
            for account in auth_accounts.split(","):
                username, password = account.split(":", 1)
                self.accounts[username] = password

    def create_token(
        self,
        username: str,
        role: str = "user",
        custom_expire_hours: int = None,
        metadata: dict = None,
    ) -> str:
        """
        Create JWT token

        Args:
            username: Username
            role: User role, default is "user", guest is "guest"
            custom_expire_hours: Custom expiration time (hours), if None use default value
            metadata: Additional metadata

        Returns:
            str: Encoded JWT token
        """
        # Choose default expiration time based on role
        if custom_expire_hours is None:
            if role == "guest":
                expire_hours = self.guest_expire_hours
            else:
                expire_hours = self.expire_hours
        else:
            expire_hours = custom_expire_hours

        expire = datetime.utcnow() + timedelta(hours=expire_hours)

        # Create payload
        payload = TokenPayload(
            sub=username, exp=expire, role=role, metadata=metadata or {}
        )

        return jwt.encode(payload.dict(), self.secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> dict:
        """
        Validate JWT token

        Args:
            token: JWT token

        Returns:
            dict: Dictionary containing user information

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            expire_timestamp = payload["exp"]
            expire_time = datetime.utcfromtimestamp(expire_timestamp)

            if datetime.utcnow() > expire_time:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            # Return complete payload instead of just username
            return {
                "username": payload["sub"],
                "role": payload.get("role", "user"),
                "metadata": payload.get("metadata", {}),
                "exp": expire_time,
            }
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )


# Get trusted API key from config
TRUSTED_API_KEY = global_args.lightrag_api_key
RATE_LIMIT_REQUESTS = global_args.rate_limit_requests
RATE_LIMIT_WINDOW = global_args.rate_limit_window
TRUSTED_NETWORKS = global_args.trusted_networks.split(',') if global_args.trusted_networks else []

# Create a list of trusted IP networks
trusted_ip_networks = [ip_network(net.strip()) for net in TRUSTED_NETWORKS if net.strip()]

def is_trusted_ip(client_ip: str) -> bool:
    """Check if an IP is in the trusted networks list."""
    if not trusted_ip_networks:
        return False
    
    try:
        client = ip_address(client_ip)
        return any(client in network for network in trusted_ip_networks)
    except ValueError:
        return False

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering forwarded headers."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # Return the first IP in the list
        return x_forwarded_for.split(',')[0].strip()
    return request.client.host if request.client else "unknown"

def is_rate_limited(client_ip: str) -> bool:
    """Check if a client IP is currently rate limited."""
    # Trusted IPs are not rate limited
    if is_trusted_ip(client_ip):
        return False
    
    current_time = time.time()
    client_data = rate_limiter.get(client_ip)
    
    # First request from this IP
    if not client_data:
        rate_limiter[client_ip] = {
            "count": 1,
            "reset_time": current_time + RATE_LIMIT_WINDOW
        }
        return False
    
    # Check if window has expired and reset if needed
    if current_time > client_data["reset_time"]:
        rate_limiter[client_ip] = {
            "count": 1,
            "reset_time": current_time + RATE_LIMIT_WINDOW
        }
        return False
    
    # Increment counter and check limit
    client_data["count"] += 1
    if client_data["count"] > RATE_LIMIT_REQUESTS:
        return True
    
    return False

async def verify_api_key(
    api_key: str = Depends(api_key_header),
    request: Request = None
) -> bool:
    """
    Verify the API key and apply rate limiting.
    
    Args:
        api_key: The API key from header
        request: FastAPI request object
        
    Returns:
        bool: True if authorized
        
    Raises:
        HTTPException: If unauthorized or rate limited
    """
    # Log attempt with client IP
    client_ip = get_client_ip(request) if request else "unknown"
    
    # Check API key
    if not api_key:
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, TRUSTED_API_KEY):
        logger.warning(f"Invalid API key attempt from {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Apply rate limiting
    if is_rate_limited(client_ip):
        reset_time = int(rate_limiter[client_ip]["reset_time"] - time.time())
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
            headers={"Retry-After": str(reset_time)}
        )
    
    return True

# Generate a secure API key if none is provided
if not TRUSTED_API_KEY:
    logger.warning("No API key configured. Generating a temporary one for this session only.")
    TRUSTED_API_KEY = secrets.token_hex(32)
    print(f"Generated temporary API key: {TRUSTED_API_KEY}")
    print("Please set a permanent API key in production.")

auth_handler = AuthHandler()
