@echo off
set "PATH=C:\LLVM_Polly\mingw64\bin;%PATH%"

if not exist build_demo mkdir build_demo
cd build_demo

echo [DEMO] Compiling Demos...
echo Copying DLL...
copy ..\build_test\libgmem.dll .

gcc ../demos/demo_generative_ram.c -I../include -L../build_test -l:libgmem.dll.a -o demo_gen_ram.exe
gcc ../demos/demo_bandwidth_efficiency.c -I../include -L../build_test -l:libgmem.dll.a -o demo_bandwidth.exe
gcc ../demos/demo_reversibility.c -I../include -L../build_test -l:libgmem.dll.a -o demo_revert.exe
gcc ../demos/demo_honeypot.c -I../include -L../build_test -l:libgmem.dll.a -o demo_honeypot.exe
gcc ../demos/demo_chaos_stability.c -I../include -L../build_test -l:libgmem.dll.a -o demo_chaos.exe
gcc ../demos/demo_ipc_coherence.c -I../include -L../build_test -l:libgmem.dll.a -o demo_ipc.exe

echo.
echo ===================================================
demo_gen_ram.exe
echo ===================================================
echo.
demo_bandwidth.exe
echo ===================================================
echo.
demo_revert.exe
echo ===================================================
echo.
demo_honeypot.exe
echo ===================================================
echo.
demo_chaos.exe
echo ===================================================
echo.
demo_ipc.exe
echo ===================================================
cd ..
