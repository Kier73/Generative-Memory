import glob
import os

print("Fixing paths in benchmarks/*.py")
files = glob.glob('benchmarks/*.py')
for pt in files:
    with open(pt, 'r') as f:
        data = f.read()
    
    # We are replacing:
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # with:
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))
    
    target = "os.path.join(os.path.dirname(__file__), '..')"
    replacement = "os.path.join(os.path.dirname(__file__), '..', 'bindings-python')"
    
    if target in data and replacement not in data:
        data = data.replace(target, replacement)
        with open(pt, 'w') as f:
            f.write(data)
        print(f"Updated {pt}")

print("Done.")
