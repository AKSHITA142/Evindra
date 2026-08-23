import sys
import os

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from tests.test_vector_retrieval import test_missing_values_retrieval, test_categorical_encoding_retrieval

if __name__ == "__main__":
    print("Running Standalone Vector Retrieval Tests...")
    print("\n--- Running Test 1: Missing Values Retrieval ---")
    test_missing_values_retrieval()
    print("\n--- Running Test 2: Categorical Encoding Retrieval ---")
    test_categorical_encoding_retrieval()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
