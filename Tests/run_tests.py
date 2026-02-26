import os
import sys
import subprocess
import time

def run_script(path):
    print(f"Running {path}...")
    start = time.time()
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    end = time.time()
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"SUBTEST PASSED ({end - start:.2f}s)\n")
        return True
    else:
        print(result.stdout)
        print(result.stderr)
        print(f"SUBTEST FAILED! ({end - start:.2f}s)\n")
        return False

def main():
    print("====================================================")
    print("   GMem v2.0.0 MASTER TEST RUNNER")
    print("====================================================\n")
    
    test_dir = os.path.dirname(__file__)
    scripts = [
        os.path.join(test_dir, "numerical/test_invariance.py"),
        os.path.join(test_dir, "statistical/test_entropy.py"),
        os.path.join(test_dir, "operational/test_reliability.py"),
        os.path.join(test_dir, "regression/test_legacy.py"),
        os.path.join(test_dir, "industry/test_numpy.py"),
        os.path.join(test_dir, "industry/test_pytorch.py"),
    ]
    
    all_passed = True
    for script in scripts:
        if not run_script(script):
            all_passed = False
    
    if all_passed:
        print("====================================================")
        print("   PASS: ALL TEST SUITES PASSED (v2.0.0 Validated)")
        print("====================================================")
        sys.exit(0)
    else:
        print("====================================================")
        print("   FAIL: TEST SUITE FAILED")
        print("====================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
