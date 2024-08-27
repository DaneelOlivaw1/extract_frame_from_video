# run_script.ps1

# 设置项目路径
$projectPath = "C:\Users\daneel\Documents\python\extract_frame_from_video"

# 设置 Conda 安装路径
$condaEnvPath = "C:\ProgramData\anaconda3"

# 设置 Python 脚本路径
$pythonScriptPath = Join-Path $projectPath "main.py"

# 设置项目特定的 Conda 环境路径
$projectCondaEnv = Join-Path $projectPath ".conda"

# 激活 Conda 环境
$activateScript = Join-Path $condaEnvPath "Scripts\activate.bat"

if (Test-Path $activateScript) {
    # 使用 cmd.exe 来运行 .bat 文件并激活环境
    cmd.exe /c "$activateScript $projectCondaEnv && python $pythonScriptPath"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to activate Conda environment or run Python script"
    }
} else {
    Write-Error "Conda activation script not found at $activateScript"
}