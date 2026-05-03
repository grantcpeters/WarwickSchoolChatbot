"""
One-time OAuth2 setup for Outlook.com IMAP access.

Run this script once to authenticate warwick-letters@outlook.com and
store the refresh token as a GitHub secret.  After that, ingest_letters.py
uses the refresh token silently — no browser interaction ever again.

Usage:
    python scripts/setup_letter_oauth.py

What it does:
1. Starts a device-code flow — prints a short URL and a code.
2. You open that URL in any browser on any device, enter the code,
   and sign in as warwick-letters@outlook.com.
3. The script receives a refresh token and saves it to:
     - GitHub secret  LETTER_OAUTH_REFRESH_TOKEN  (for Actions)
     - .env           LETTER_OAUTH_REFRESH_TOKEN   (for local use)
"""

import os
import re
import sys

import msal
from dotenv import load_dotenv

load_dotenv()

# Azure AD app registered with PersonalMicrosoftAccount audience.
# Created via: az ad app create --display-name "WarwickLetterIngest"
#   --sign-in-audience "PersonalMicrosoftAccount"
#   --public-client-redirect-uris "https://login.microsoftonline.com/common/oauth2/nativeclient"
CLIENT_ID = os.getenv("LETTER_OAUTH_CLIENT_ID", "d983cb90-69c5-4d9c-b6b2-cd2a413c241a")

# consumers = personal Microsoft accounts only (Outlook.com / Hotmail / Live)
AUTHORITY = "https://login.microsoftonline.com/consumers"

SCOPES = [
    "https://graph.microsoft.com/Mail.ReadWrite",
]

ENV_FILE = os.path.join(os.path.dirname(__file__), "..", ".env")


def _update_env_file(key: str, value: str) -> None:
    """Write or replace KEY=value in the local .env file."""
    path = os.path.abspath(ENV_FILE)
    try:
        with open(path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    pattern = rf"^{re.escape(key)}=.*$"
    new_line = f"{key}={value}"
    if re.search(pattern, text, re.MULTILINE):
        text = re.sub(pattern, new_line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n{new_line}\n"

    with open(path, "w") as f:
        f.write(text)
    print(f"  .env updated: {key}=<saved>")


def main() -> None:
    print("=" * 60)
    print("Warwick Letter Ingest — OAuth2 one-time setup")
    print("=" * 60)
    print(f"Client ID : {CLIENT_ID}")
    print(f"Authority : {AUTHORITY}")
    print()

    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        print("ERROR: Failed to start device flow:")
        print(flow.get("error_description", flow))
        sys.exit(1)

    # Print the message exactly — it contains the URL and code
    print(flow["message"])
    print()
    print("Sign in as warwick-letters@outlook.com when prompted.")
    print("Waiting for authentication ...")
    print()

    result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        print("ERROR: Authentication failed:")
        print(result.get("error_description", result))
        sys.exit(1)

    refresh_token = result.get("refresh_token")
    if not refresh_token:
        print(
            "ERROR: No refresh token in response. Make sure offline_access was granted."
        )
        sys.exit(1)

    print("Authentication successful!")
    print()

    # Save to GitHub secrets
    import subprocess

    print("Saving to GitHub secrets ...")
    try:
        subprocess.run(
            [
                "gh",
                "secret",
                "set",
                "LETTER_OAUTH_REFRESH_TOKEN",
                "--body",
                refresh_token,
            ],
            check=True,
        )
        print("  ✓ GitHub secret LETTER_OAUTH_REFRESH_TOKEN saved")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Could not save to GitHub secrets: {e}")
        print(f"  Refresh token (save this manually): {refresh_token}")

    # Save to local .env
    _update_env_file("LETTER_OAUTH_REFRESH_TOKEN", refresh_token)

    print()
    print("Setup complete. You can now run scripts/ingest_letters.py")


if __name__ == "__main__":
    main()
