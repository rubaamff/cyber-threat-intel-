# Script for extracting Indicators of Compromise (IOCs)

def extract_iocs(text_data):
    """
    Placeholder function to extract IOCs from text data.
    In a real scenario, this would involve regex, pattern matching, or NLP techniques
    to identify IPs, URLs, hashes, etc.
    """
    iocs = {
        "ips": [],
        "urls": [],
        "hashes": []
    }
    # Example: Simple regex for IP addresses (for demonstration purposes)
    import re
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    found_ips = re.findall(ip_pattern, text_data)
    iocs["ips"].extend(found_ips)

    return iocs

if __name__ == '__main__':
    sample_text = "Detected suspicious activity from 192.168.1.100 and accessed http://malicious.com/payload.exe. File hash: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
    extracted = extract_iocs(sample_text)
    print("Extracted IOCs:", extracted)


