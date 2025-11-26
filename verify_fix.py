
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from utils.html_report_generator import HTMLReportGenerator

def verify_fix():
    print("Verifying HTMLReportGenerator.save_report fix...")
    try:
        generator = HTMLReportGenerator()
        if hasattr(generator, 'save_report'):
            print("SUCCESS: HTMLReportGenerator has 'save_report' attribute.")
            
            # Try to save a dummy report
            dummy_content = "<html><body><h1>Test Report</h1></body></html>"
            output_path = "test_report_verification.html"
            
            if generator.save_report(dummy_content, output_path):
                print(f"SUCCESS: Report saved to {output_path}")
                # Clean up
                if os.path.exists(output_path):
                    os.remove(output_path)
                    print("Cleaned up test file.")
            else:
                print("FAILURE: save_report returned False.")
        else:
            print("FAILURE: HTMLReportGenerator still missing 'save_report' attribute.")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_fix()
