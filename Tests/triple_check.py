import os
import sys
import subprocess
import time

# List of critical test suites to run in the triple-check
TEST_SCRIPTS = [
    "Tests/audit_integrity.py",
    "Tests/adversarial_gauntlet.py",
    "Tests/stress_multithreading.py",
    "Tests/stress_recursion.py",
    "Tests/stress_collision_gauntlet.py",
    "Tests/stress_algebraic.py",
    "Tests/stress_aof_massive.py",
    "Tests/functional_pytorch.py",
    "Tests/test_layering.py",
    "Tests/verify_precision.py"
]

def run_triple_check():
    print("====================================================")
    print("   GMEM FINAL VALIDATION: TRIPLE CHECK RUN")
    print("====================================================\n")
    
    overall_success = True
    for i in range(1, 4):
        print(f"--- PASS {i} STARTING ---")
        pass_success = True
        for script in TEST_SCRIPTS:
            print(f"  Running {script}...", end="", flush=True)
            try:
                # Use sys.executable to ensure we use the same python interpreter
                res = subprocess.run([sys.executable, script], 
                                     capture_output=True, text=True, timeout=300)
                if res.returncode == 0:
                    print(" [OK]")
                else:
                    print(" [FAILED]")
                    print(res.stderr)
                    pass_success = False
            except Exception as e:
                print(f" [ERROR] {e}")
                pass_success = False
        
        if not pass_success:
            print(f"--- PASS {i} FAILED ---")
            overall_success = False
        else:
            print(f"--- PASS {i} PASSED ---")
        print("-" * 50)
        
    return overall_success

if __name__ == "__main__":
    if run_triple_check():
        print("\nTRIPLE CHECK COMPLETE: ALL TESTS STABLE.")
        sys.exit(0)
    else:
        print("\nTRIPLE CHECK FAILED: INSTABILITY DETECTED.")
        sys.exit(1)
