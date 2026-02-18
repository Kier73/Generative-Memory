@echo off
set "PATH=C:\LLVM_Polly\mingw64\bin;%PATH%"

echo [TEST RUNNER] Cleaning previous build...
if exist build_test rmdir /s /q build_test
mkdir build_test
cd build_test

echo [TEST RUNNER] Configuring CMake...
cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..
if %errorlevel% neq 0 exit /b %errorlevel%

echo [TEST RUNNER] Building Codebase...
cmake --build .
if %errorlevel% neq 0 exit /b %errorlevel%

echo [TEST RUNNER] Compiling Tests...
gcc ../tests/test_alloc_safe.c -I../include -L. -lgmem -o test_alloc_safe.exe
if %errorlevel% neq 0 exit /b %errorlevel%

gcc ../tests/test_aro_ast.c -I../include -L. -lgmem -o test_aro_ast.exe
if %errorlevel% neq 0 exit /b %errorlevel%

gcc ../tests/test_archetype_dyn.c -I../include -L. -lgmem -o test_archetype_dyn.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [TEST RUNNER] Running Tests...
echo ---------------------------------------------------
test_alloc_safe.exe
if %errorlevel% neq 0 (
    echo [FAIL] test_alloc_safe failed
    exit /b 1
)

test_aro_ast.exe
if %errorlevel% neq 0 (
    echo [FAIL] test_aro_ast failed
    exit /b 1
)

test_archetype_dyn.exe
if %errorlevel% neq 0 (
    echo [FAIL] test_archetype_dyn failed
    exit /b 1
)

echo ---------------------------------------------------
echo [SUCCESS] All Systems Hardened and Verified.
exit /b 0
