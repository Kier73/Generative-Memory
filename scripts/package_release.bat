@echo off
set "PATH=C:\LLVM_Polly\mingw64\bin;%PATH%"

echo [PKG] Packaging Release...

if exist release rmdir /s /q release
mkdir release
mkdir release\bin
mkdir release\include
mkdir release\lib
mkdir release\bindings
mkdir release\demos

echo [PKG] Building Release Configuration...
if not exist build_release mkdir build_release
cd build_release
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-mavx2" ..
cmake --build . --config Release
cd ..

echo [PKG] Copying Artifacts...
copy build_release\libgmem.dll release\bin\
copy build_release\libgmem.dll.a release\lib\
copy include\*.h release\include\
copy README.md release\
copy LICENSE* release\

echo [PKG] Copying Bindings...
xcopy bindings release\bindings /s /e /y

echo [PKG] Copying Demos (Source)...
copy demos\*.c release\demos\
copy demos\run_demos.bat release\demos\

echo [PKG] Verifying Package...
if exist release\bin\libgmem.dll (
    echo [SUCCESS] Package created in 'release/'
) else (
    echo [FAIL] Build failed.
)
exit /b 0
