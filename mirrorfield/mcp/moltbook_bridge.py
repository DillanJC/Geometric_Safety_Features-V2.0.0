"""
Moltbook REST API bridge for posting with confidence metadata.

Uses the Moltbook API (https://www.moltbook.com) directly via urllib — no
third-party SDK needed. Reads MOLTBOOK_API_KEY and optionally MOLTBOOK_API_URL
from the environment. All functions degrade gracefully when the API key is not
configured.

Moltbook requires a verification challenge (math problem) after creating posts
and comments. This bridge solves the challenge automatically so content is
published immediately.
"""

import json
import os
import re
import urllib.request
import urllib.error

MOLTBOOK_DEFAULT_URL = "https://www.moltbook.com/api/v1"


def _api_url() -> str:
    return os.environ.get("MOLTBOOK_API_URL", MOLTBOOK_DEFAULT_URL).rstrip("/")


def _make_request(url: str, api_key: str, data: dict | None = None,
                  method: str = "GET") -> dict:
    """Make an authenticated request to the Moltbook API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=payload, headers=headers,
                                method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000,
}


def _parse_word_number(words: list[str]) -> list[float]:
    """Parse English word numbers from a list of words.

    Handles compound forms like "thirty seven" -> 37, "one hundred" -> 100,
    "two hundred fifty three" -> 253.
    """
    results: list[float] = []
    current = 0.0
    in_number = False

    for w in words:
        if w in _WORD_NUMBERS:
            val = _WORD_NUMBERS[w]
            if val == 100:
                current = (current if current else 1) * 100
            elif val == 1000:
                current = (current if current else 1) * 1000
            else:
                current += val
            in_number = True
        else:
            if in_number:
                results.append(current)
                current = 0.0
                in_number = False
    if in_number:
        results.append(current)

    return results


def _solve_challenge(challenge: str) -> str:
    """Solve a Moltbook verification math challenge.

    Challenges are obfuscated text describing addition/subtraction of forces
    in newtons. Extract the numbers and operations, compute, return as "X.XX".
    """
    # Flatten the obfuscated casing/punctuation
    clean = re.sub(r"[^a-zA-Z0-9.,;? ]", "", challenge).lower()

    # Extract digit numbers first
    numbers = [float(x) for x in re.findall(r"[\d]+(?:\.[\d]+)?", clean)]

    # If not enough digit numbers, try word numbers
    if len(numbers) < 2:
        words = clean.split()
        word_nums = _parse_word_number(words)
        numbers = word_nums if len(word_nums) >= 2 else numbers + word_nums

    if len(numbers) < 2:
        return f"{numbers[0]:.2f}" if numbers else "0.00"

    # Typical pattern: starts with X, increases/decreases by Y
    if "increases" in clean or "adds" in clean or "plus" in clean:
        result = numbers[0] + numbers[1]
    elif "decreases" in clean or "minus" in clean or "subtracts" in clean:
        result = numbers[0] - numbers[1]
    else:
        # Default to addition (most common challenge type)
        result = numbers[0] + numbers[1]

    return f"{result:.2f}"


def _auto_verify(body: dict, api_key: str) -> dict | None:
    """If the response requires verification, solve and submit it.

    Returns the verify response on success, or None if no verification needed.
    """
    verification = body.get("verification")
    if not verification:
        return None

    code = verification.get("code", "")
    challenge = verification.get("challenge", "")

    if not code or not challenge:
        return None

    answer = _solve_challenge(challenge)

    verify_data = {
        "verification_code": code,
        "answer": answer,
    }

    try:
        return _make_request(
            f"{_api_url()}/verify", api_key, data=verify_data, method="POST"
        )
    except Exception:
        return None


def is_moltbook_configured() -> bool:
    """Return True if a Moltbook API key is present in the environment."""
    return bool(os.environ.get("MOLTBOOK_API_KEY"))


def post_to_moltbook(
    submolt: str,
    title: str,
    content: str,
    confidence_score: float,
    confidence_label: str,
    metrics: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """Post content to Moltbook with embedded agent-metadata block.

    Uses ``POST /posts`` with Bearer auth per the Moltbook API spec.
    Automatically solves the verification challenge if required.

    Returns a dict with ``ok``, ``post_id`` on success,
    or ``ok=False`` and an ``error`` message on failure.
    """
    api_key = api_key or os.environ.get("MOLTBOOK_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": (
                "MOLTBOOK_API_KEY not set. Configure it in the environment "
                "to enable Moltbook integration."
            ),
        }

    # Build metadata block
    meta = {
        "confidence_score": round(confidence_score, 4),
        "confidence_label": confidence_label,
    }
    if metrics:
        meta["metrics"] = metrics

    metadata_block = (
        "\n---\n```agent-metadata\n"
        + json.dumps(meta, indent=2)
        + "\n```"
    )
    full_content = content + metadata_block

    data = {
        "submolt": submolt,
        "title": title,
        "content": full_content,
    }

    try:
        body = _make_request(f"{_api_url()}/posts", api_key, data=data,
                             method="POST")

        # Extract post_id from nested response
        post_obj = body.get("post", body)
        post_id = post_obj.get("id")

        # Auto-verify if needed
        verified = _auto_verify(body, api_key)

        result = {"ok": True, "post_id": post_id}
        if verified:
            result["verified"] = verified.get("success", False)
        return result
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode()
        except Exception:
            err_body = str(exc)
        return {
            "ok": False,
            "error": f"Moltbook API error {exc.code}: {err_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Moltbook post failed: {exc}",
        }


def comment_on_moltbook(
    post_id: str,
    content: str,
    confidence_score: float | None = None,
    confidence_label: str | None = None,
    parent_id: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Add a comment to a Moltbook post via ``POST /posts/:id/comments``.

    Optionally appends an agent-metadata block if confidence info is provided.
    Supports nested replies via *parent_id*. Automatically solves the
    verification challenge if required.
    """
    api_key = api_key or os.environ.get("MOLTBOOK_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": (
                "MOLTBOOK_API_KEY not set. Configure it in the environment "
                "to enable Moltbook integration."
            ),
        }

    full_content = content
    if confidence_score is not None and confidence_label is not None:
        meta = {
            "confidence_score": round(confidence_score, 4),
            "confidence_label": confidence_label,
        }
        full_content += (
            "\n---\n```agent-metadata\n"
            + json.dumps(meta, indent=2)
            + "\n```"
        )

    data: dict = {"content": full_content}
    if parent_id is not None:
        data["parent_id"] = parent_id

    try:
        body = _make_request(
            f"{_api_url()}/posts/{post_id}/comments", api_key, data=data,
            method="POST"
        )

        # Extract comment_id from nested response
        comment_obj = body.get("comment", body)
        comment_id = comment_obj.get("id")

        # Auto-verify if needed
        verified = _auto_verify(body, api_key)

        result = {"ok": True, "comment_id": comment_id}
        if verified:
            result["verified"] = verified.get("success", False)
        return result
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode()
        except Exception:
            err_body = str(exc)
        return {
            "ok": False,
            "error": f"Moltbook API error {exc.code}: {err_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Moltbook comment failed: {exc}",
        }
