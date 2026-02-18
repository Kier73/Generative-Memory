# Contributing to Generative Memory

Generating things that generate

## Getting Started

1.  **Fork the repository**: Click the "Fork" button on the top right of this page.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/Kier73/Generative-Memory.git
    cd generative_memory
    ```
3.  **Create a branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```

## Development Workflow

### Building locally

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Running Tests

Please ensure all tests pass before submitting a pull request.

**C Tests (Gauntlet):**
These tests stress the memory system. Run them from the project root.
```bash
python tests/gauntlet_exascale_stress.py
python tests/gauntlet_material_stress.py
```

**Unit Tests:**
Run standard CTests if available:
```bash
cd build
ctest
```

## Pull Request Process

1.  Ensure your code adheres to the existing coding style (C99).
2.  Update documentation if you are changing functionality.
3.  Add tests for any new features or bug fixes.
4.  Submit a Pull Request to the `main` branch.
5.  Provide a clear description of your changes.


