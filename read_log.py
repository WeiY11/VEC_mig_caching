
import sys

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('utf-8', 'replace').decode('utf-8'))

try:
    with open('output_cmd_verify_7.txt', 'rb') as f:
        content = f.read()
        # Try decoding with different encodings
        decoded = False
        for encoding in ['utf-8', 'gbk', 'utf-16', 'cp1252']:
            try:
                text = content.decode(encoding)
                safe_print(f"--- Decoded with {encoding} ---")
                safe_print(text[:2000])
                decoded = True
                break
            except UnicodeDecodeError:
                continue
        
        if not decoded:
            print("Could not decode file with common encodings.")
            print(content[:1000]) # Print first 1000 bytes raw
except Exception as e:
    print(f"Error: {e}")
