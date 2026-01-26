# test_services.py - FIXED IMPORT
import sys
import os

print("=" * 60)
print("Testing AI Fake News Detector Services")
print("=" * 60)

# MANUALLY add services to path for testing from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services'))

# Test 1: Import services
print("\n1. Testing imports...")
try:
    # Try different import patterns
    import services
    from services import DetectionOrchestrator
    
    print("✓ Successfully imported services")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    # Try direct import as fallback
    try:
        from services.orchestrator import DetectionOrchestrator
        from services.inference import InferenceService
        print("✓ Fallback import successful")
    except ImportError as e2:
        print(f"✗ Fallback also failed: {e2}")
        sys.exit(1)