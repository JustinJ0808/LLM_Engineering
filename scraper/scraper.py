import requests
from bs4 import BeautifulSoup
import json
import re

def clean_text(text):
    """
    Cleans up the extracted text: removes extra whitespace, 
    new lines, and non-printable characters.
    """
    # Replace multiple newlines/whitespace with a single newline or space
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading/trailing whitespace from each line and join
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def scrape_for_llm(url, char_limit=2000):
    """
    Scrapes a website, extracts main text content, and truncates it.
    Returns a dictionary suitable for LLM training data.
    """
    try:
        # User-agent to avoid getting blocked by some sites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements as well as common boilerplate
        ignore_tags = ["script", "style", "header", "footer", "nav", "aside", "form", "button", "svg", "noscript"]
        for tag in soup(ignore_tags):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=' ')
        
        # Clean text
        text = clean_text(text)
        
        # Truncate
        if len(text) > char_limit:
            text = text[:char_limit] + "..."
            
        return {
            "url": url,
            "title": soup.title.string if soup.title else "No Title",
            "content": text
        }

    except Exception as e:
        return {"url": url, "error": str(e)}

if __name__ == "__main__":
    # Example usage
    target_url = "https://en.wikipedia.org/wiki/Large_language_model"
    print(f"Scraping: {target_url} ...")
    
    result = scrape_for_llm(target_url)
    
    if "error" in result:
        print(f"Failed to scrape: {result['error']}")
    else:
        # Output as JSON (common for LLM training datasets like JSONL)
        print("\n--- Extracted Data (Truncated to 2000 chars) ---")
        print(json.dumps(result, indent=2))
        
        # Optionally save to a file
        with open("training_data.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print("\nData appended to training_data.jsonl")
