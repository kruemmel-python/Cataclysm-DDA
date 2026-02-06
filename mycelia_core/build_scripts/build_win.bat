@echo off
echo Building Mycelia Core DLL...

:: Erstelle build ordner falls nicht vorhanden
if not exist "../bin" mkdir "../bin"

:: Der GCC Befehl
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS -DMYCELIA_EXPORTS -shared ../src/mycelia_core.c ../src/CipherCore_NoiseCtrl.c -o ../bin/CC_OpenCl.dll -I"../include" -I"../src" -I"../CL" -L"../CL" -lOpenCL "-Wl,--out-implib,../lib/libCC_OpenCl.a" -static-libstdc++ -static-libgcc
echo Build complete. Check /bin folder.
 gcc -o ../bin/test_sdk.exe ../samples/test_sdk.c ../lib/libCC_OpenCl.a -I"../include" -mconsole
echo Build complete. Check /bin folder.
pause