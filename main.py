import modal

# Basic Modal app for triage-ai-backend
app = modal.App("triage-ai-backend")

# Simple image with basic Python packages
image = modal.Image.debian_slim().pip_install(["requests"])


@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    
    # Test hello function
    hello_result = hello_triage.remote()
    print(f"Hello result: {hello_result}")
    
    # Test data processing
    test_data = {"test": "data", "number": 42}
    process_result = process_data.remote(test_data)
    print(f"Process result: {process_result}")
    
    print("✅ All tests completed!")