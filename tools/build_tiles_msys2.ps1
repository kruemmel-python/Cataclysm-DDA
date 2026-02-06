param(
    [int]$Jobs = 2,
    [string]$Msys2Root = "",
    [string]$Target = "cataclysm-tiles.exe"
)

$ErrorActionPreference = "Stop"

if ($null -eq $Jobs -or $Jobs -lt 1) {
    $Jobs = 1
}

function Resolve-Msys2RootPath {
    param(
        [string]$Root
    )

    if ($Root) {
        if (Test-Path $Root) {
            return (Resolve-Path $Root).Path
        }
    }

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:MSYS2_ROOT) {
        $candidates.Add($env:MSYS2_ROOT)
    }
    $candidates.Add("C:\\msys64")
    $candidates.Add("C:\\tools\\msys64")
    $candidates.Add("D:\\msys64")
    $candidates.Add("C:\\msys2")
    $candidates.Add("C:\\msys2\\msys64")

    foreach ($cand in $candidates) {
        if (-not $cand) {
            continue
        }
        $root = $cand.TrimEnd("\", "/")
        $probe = Join-Path $root "mingw64\\bin\\pkg-config.exe"
        if (Test-Path $probe) {
            return (Resolve-Path $root).Path
        }
    }

    $bash = Get-Command bash.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bash) {
        $maybeRoot = Resolve-Path (Join-Path $bash.Source "..\\..") -ErrorAction SilentlyContinue
        if ($maybeRoot) {
            $probe = Join-Path $maybeRoot.Path "mingw64\\bin\\pkg-config.exe"
            if (Test-Path $probe) {
                return $maybeRoot.Path
            }
        }
    }

    throw "MSYS2 root not found. Provide -Msys2Root C:\\msys64"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$msys2RootPath = Resolve-Msys2RootPath -Root $Msys2Root
$mingwBin = Join-Path $msys2RootPath "mingw64\bin"
$usrBin = Join-Path $msys2RootPath "usr\bin"
$pkgConfig = Join-Path $mingwBin "pkg-config.exe"
$objdump = Join-Path $mingwBin "objdump.exe"
$strip = Join-Path $mingwBin "strip.exe"

if (-not (Test-Path $mingwBin)) {
    throw "Mingw bin folder not found: $mingwBin"
}
if (-not (Test-Path $pkgConfig)) {
    throw "pkg-config not found: $pkgConfig"
}
if (-not (Test-Path $objdump)) {
    throw "objdump not found: $objdump"
}

$pkgSysInc = ($mingwBin -replace "\\", "/") + "/../include"

$env:PATH = "$mingwBin;$usrBin;$env:PATH"
$env:PKG_CONFIG = $pkgConfig
$env:PKG_CONFIG_SYSTEM_INCLUDE_PATH = $pkgSysInc
$env:TEMP = (Join-Path $repoRoot ".tmp")
$env:TMP = (Join-Path $repoRoot ".tmp")
$env:TMPDIR = (Join-Path $repoRoot ".tmp")
$env:GIT_CONFIG_GLOBAL = (Join-Path $repoRoot ".gitconfig_temp")
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

Push-Location $repoRoot
try {
    $baseArgs = @("NATIVE=win32", "LOCALIZE=0", "TILES=1", "DYNAMIC_LINKING=1", "TESTS=0")
    $buildSucceeded = $false
    $jobsToTry = New-Object System.Collections.Generic.List[int]
    $jobsToTry.Add([Math]::Max(1, $Jobs))
    if ($Jobs -gt 1) {
        $jobsToTry.Add(1)
    }

    foreach ($jobCount in $jobsToTry) {
        $makeArgs = @()
        $makeArgs += $baseArgs
        if ($jobCount -ge 1) {
            $makeArgs += "-j"
            $makeArgs += "$jobCount"
        }
        if ($jobCount -eq 1 -and $Jobs -gt 1) {
            # Retry mode for low-memory systems.
            $makeArgs += "PCH=0"
            $makeArgs += "OPTLEVEL=-O0"
        }
        $makeArgs += $Target

        if ($jobCount -eq 1 -and $Jobs -gt 1) {
            Write-Warning "Retry build with low-memory settings (-j1, PCH=0, OPTLEVEL=-O0)..."
        }

        & make @makeArgs
        if ($LASTEXITCODE -eq 0) {
            $buildSucceeded = $true
            break
        }

        if ($jobCount -ne 1 -or $Jobs -le 1) {
            Write-Warning "Build failed with exit code $LASTEXITCODE (jobs=$jobCount)."
        }
    }

    if (-not $buildSucceeded) {
        throw "Build failed after retries."
    }

    $targetSrc = Join-Path $repoRoot $Target
    if ( ( Test-Path $strip ) -and ( Test-Path $targetSrc ) -and $Target.ToLowerInvariant().EndsWith(".exe") ) {
        # Reduce peak disk usage: strip build output before any dist copy.
        & $strip --strip-debug $targetSrc
    }

    $distDir = Join-Path $repoRoot "dist"
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    $targetDist = Join-Path $distDir $Target
    $targetDistNewName = if ($Target.ToLowerInvariant().EndsWith(".exe")) {
        $Target.Substring(0, $Target.Length - 4) + ".new.exe"
    } else {
        "$Target.new"
    }
    $targetDistNew = Join-Path $distDir $targetDistNewName
    try {
        Copy-Item $targetSrc -Destination $targetDist -Force -ErrorAction Stop
    } catch {
        Write-Warning "Could not overwrite $targetDist (likely locked). Writing $targetDistNew instead."
        Copy-Item $targetSrc -Destination $targetDistNew -Force -ErrorAction Stop
    }
    $distBin = Join-Path $distDir "bin"
    New-Item -ItemType Directory -Force -Path $distBin | Out-Null
    $distUser = Join-Path $distDir "user"
    New-Item -ItemType Directory -Force -Path $distUser | Out-Null

    $copyDirs = @(
        "data",
        "gfx",
        "mods",
        "sound",
        "font",
        "lang",
        "config",
        "templates",
        "doc",
        "achievements",
        "bin"
    )
    foreach ($dir in $copyDirs) {
        $src = Join-Path $repoRoot $dir
        if (Test-Path $src) {
            Copy-Item -Path $src -Destination $distDir -Recurse -Force
        }
    }

    $licenseFiles = Get-ChildItem -Path $repoRoot -Filter "LICENSE*.txt" -File -ErrorAction SilentlyContinue
    foreach ($lf in $licenseFiles) {
        Copy-Item -Path $lf.FullName -Destination $distDir -Force
    }
    $copyFiles = @("README.md", "DNA_README.md")
    foreach ($file in $copyFiles) {
        $srcFile = Join-Path $repoRoot $file
        if (Test-Path $srcFile) {
            Copy-Item -Path $srcFile -Destination $distDir -Force
        }
    }

    $queue = New-Object System.Collections.Generic.Queue[string]
    if (Test-Path $targetDist) {
        $queue.Enqueue($targetDist)
    } elseif (Test-Path $targetDistNew) {
        $queue.Enqueue($targetDistNew)
    }
    $seen = @{}

    while ($queue.Count -gt 0) {
        $file = $queue.Dequeue()
        if (-not (Test-Path $file)) {
            continue
        }
        $deps = & $objdump -p $file 2>$null | Select-String -Pattern "DLL Name" | ForEach-Object {
            ($_ -split "DLL Name:")[1].Trim()
        }
        foreach ($dep in $deps) {
            if ($seen.ContainsKey($dep)) {
                continue
            }
            $seen[$dep] = $true
            $depPath = Join-Path $mingwBin $dep
            if (Test-Path $depPath) {
                Copy-Item $depPath -Destination $distDir -Force
                $queue.Enqueue($depPath)
            }
        }
    }

    if (Test-Path $strip) {
        $stripTargets = @()
        if (Test-Path $targetDist) {
            $stripTargets += $targetDist
        }
        if (Test-Path $targetDistNew) {
            $stripTargets += $targetDistNew
        }
        foreach ($stripTarget in $stripTargets) {
            & $strip --strip-debug $stripTarget
        }
    }

    $batPath = Join-Path $distDir "run-cataclysm-tiles.bat"
    @"
@echo off
setlocal
  set "DIST=%~dp0"
  set "DIST_NOSLASH=%DIST:~0,-1%"
  set "DATA_DIR=%DIST%data"
  set "USER_DIR=%DIST%user"
  set "BIN_DIR=%DIST%bin"
  set "PATH=%BIN_DIR%;%PATH%"
  set "EXE=%DIST%$Target"
  if exist "%DIST%$targetDistNewName" (
    set "EXE=%DIST%$targetDistNewName"
  )

if not exist "%DATA_DIR%\" (
  echo [ERROR] Missing data directory: %DATA_DIR%
  exit /b 1
)
if not exist "%EXE%" (
  echo [ERROR] Missing executable: %EXE%
  exit /b 1
)
if not exist "%USER_DIR%\" (
  echo [INFO] Creating user directory: %USER_DIR%
  mkdir "%USER_DIR%" >nul 2>&1
)
if not exist "%BIN_DIR%\CC_OpenCl.dll" (
  echo [WARN] CC_OpenCl.dll not found in %BIN_DIR%\ (SubQG will use fallback seed)
)

pushd "%DIST%"
  "%EXE%" --basepath "%DIST_NOSLASH%" --userdir "%USER_DIR%" %*
popd
endlocal
"@ | Set-Content -Path $batPath -Encoding ASCII

    $subqgCandidates = @(
        (Join-Path $repoRoot "mycelia_core\\bin\\CC_OpenCl.dll"),
        (Join-Path $repoRoot "bin\\CC_OpenCl.dll")
    )
    $subqgDll = $null
    foreach ($cand in $subqgCandidates) {
        if (Test-Path $cand) {
            $subqgDll = $cand
            break
        }
    }
    if ($subqgDll) {
        Copy-Item $subqgDll -Destination (Join-Path $distBin "CC_OpenCl.dll") -Force
        Copy-Item $subqgDll -Destination (Join-Path $distDir "CC_OpenCl.dll") -Force
    } else {
        Write-Warning "CC_OpenCl.dll not found in mycelia_core\\bin or bin."
    }
}
finally {
    Pop-Location
}
