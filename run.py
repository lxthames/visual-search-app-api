#!/usr/bin/env python3
"""
Simple script to run the FastAPI application.
Just run: python run.py
"""
import uvicorn
from pycloudflared import try_cloudflare  # <--- 1. Import this

if __name__ == "__main__":
    # <--- 2. Start the tunnel BEFORE starting the server
    # This automatically downloads 'cloudflared' if needed and gives you a public URL
    tunnel_url = try_cloudflare(port=8000) 
    print(f"Public Cloudflare URL: {tunnel_url}") # <--- 3. Print the link

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )