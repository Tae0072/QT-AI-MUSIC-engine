# =====================================================================
#  YuE 노래 생성기 - 로딩바 + 실행 + 자동 종료 스크립트
#  역할:
#   (1) 서버(python)를 백그라운드로 시작
#   (2) 화면 상단 중앙에 진행바 표시
#   (3) 준비되면 "전용 앱 창"(Chrome app 모드)으로 앱을 엶
#   (4) 그 앱 창을 닫으면 백그라운드 서버도 함께 종료 (watchdog)
#  ※ 보통 YuE_실행.vbs 가 이 파일을 자동 실행합니다.
# =====================================================================
$ErrorActionPreference = "SilentlyContinue"

# --- 경로/설정 ---------------------------------------------------------
$base = $PSScriptRoot
if (-not $base) { $base = Split-Path -Parent $MyInvocation.MyCommand.Path }
$venvPy = Join-Path $base "venv310\Scripts\python.exe"
$work   = Join-Path $base "inference"
$script:url         = "http://localhost:7862/"
$port               = 7862
$script:expectedSec = 90
$script:softMsgSec  = 240
$appProfile = Join-Path $env:TEMP ("yue_app_" + $PID)   # 실행마다 고유 프로필(중복 실행 충돌 방지)

# --- 사용할 브라우저 찾기(앱 모드 지원: Chrome > Edge > Whale) ----------
function Find-Browser {
    $cands = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LocalAppData\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Naver\Naver Whale\Application\whale.exe",
        "${env:ProgramFiles(x86)}\Naver\Naver Whale\Application\whale.exe",
        "$env:LocalAppData\Naver\Naver Whale\Application\whale.exe"
    )
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

# --- 포트 준비 확인(빠른 TCP 연결 시도) --------------------------------
function Test-Ready {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $iar = $c.BeginConnect("127.0.0.1", $port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(700, $false)
        $connected = ($ok -and $c.Connected)
        $c.Close()
        return $connected
    } catch { return $false }
}

# --- 서버 종료(7862 포트 소유 프로세스 트리 + 우리가 띄운 PID) ----------
function Stop-Server {
    $owners = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
    foreach ($o in $owners) { if ($o) { & taskkill.exe /PID $o /T /F 2>$null | Out-Null } }
    if ($script:serverPid) { & taskkill.exe /PID $script:serverPid /T /F 2>$null | Out-Null }
}

$browser = Find-Browser
$script:serverPid = $null
$script:ready     = $false
$script:cancelled = $false

# --- 서버가 이미 떠 있으면 바로 앱 열기, 아니면 시작 + 로딩바 ----------
if (Test-Ready) {
    $script:ready = $true
}
else {
    $env:PYTHONUTF8 = "1"
    $env:HF_HUB_DISABLE_XET = "1"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $sp = Start-Process -FilePath $venvPy -ArgumentList 'gradio_server.py --profile 3 --sdpa --server_port 7862' -WorkingDirectory $work -WindowStyle Hidden -PassThru
    if ($sp) { $script:serverPid = $sp.Id }

    # ----- 로딩바 UI(WinForms) -----
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    $W = 520; $H = 104
    $script:form = New-Object System.Windows.Forms.Form
    $script:form.FormBorderStyle = 'None'
    $script:form.StartPosition   = 'Manual'
    $script:form.Size            = New-Object System.Drawing.Size($W, $H)
    $area = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $script:form.Location = New-Object System.Drawing.Point([int](($area.Width - $W)/2), 14)
    $script:form.TopMost       = $true
    $script:form.ShowInTaskbar = $true
    $script:form.BackColor     = [System.Drawing.Color]::FromArgb(28, 28, 38)
    $script:form.Text          = "YuE 노래 생성기 준비 중"

    $rad = 18
    $gp = New-Object System.Drawing.Drawing2D.GraphicsPath
    $gp.AddArc(0, 0, $rad, $rad, 180, 90)
    $gp.AddArc($W-$rad, 0, $rad, $rad, 270, 90)
    $gp.AddArc($W-$rad, $H-$rad, $rad, $rad, 0, 90)
    $gp.AddArc(0, $H-$rad, $rad, $rad, 90, 90)
    $gp.CloseAllFigures()
    $script:form.Region = New-Object System.Drawing.Region($gp)

    $fontTitle = New-Object System.Drawing.Font("Malgun Gothic", 11, [System.Drawing.FontStyle]::Bold)
    $fontBody  = New-Object System.Drawing.Font("Malgun Gothic", 9)

    $script:title = New-Object System.Windows.Forms.Label
    $script:title.Text      = [char]0x266A + "  AI 노래 생성기를 준비하고 있어요"
    $script:title.Font      = $fontTitle
    $script:title.ForeColor = [System.Drawing.Color]::White
    $script:title.AutoSize  = $false
    $script:title.Size      = New-Object System.Drawing.Size(($W-60), 24)
    $script:title.Location  = New-Object System.Drawing.Point(20, 14)
    $script:title.BackColor = [System.Drawing.Color]::Transparent
    $script:form.Controls.Add($script:title)

    $script:status = New-Object System.Windows.Forms.Label
    $script:status.Text      = "모델을 불러오는 중입니다 (최대 1~2분)"
    $script:status.Font      = $fontBody
    $script:status.ForeColor = [System.Drawing.Color]::FromArgb(180, 185, 200)
    $script:status.AutoSize  = $false
    $script:status.Size      = New-Object System.Drawing.Size(($W-120), 20)
    $script:status.Location  = New-Object System.Drawing.Point(20, 40)
    $script:status.BackColor = [System.Drawing.Color]::Transparent
    $script:form.Controls.Add($script:status)

    $script:elapsed = New-Object System.Windows.Forms.Label
    $script:elapsed.Text      = "0초"
    $script:elapsed.Font      = $fontBody
    $script:elapsed.ForeColor = [System.Drawing.Color]::FromArgb(140, 145, 160)
    $script:elapsed.AutoSize  = $false
    $script:elapsed.TextAlign = 'MiddleRight'
    $script:elapsed.Size      = New-Object System.Drawing.Size(80, 20)
    $script:elapsed.Location  = New-Object System.Drawing.Point(($W-100), 40)
    $script:elapsed.BackColor = [System.Drawing.Color]::Transparent
    $script:form.Controls.Add($script:elapsed)

    $script:bar = New-Object System.Windows.Forms.ProgressBar
    $script:bar.Style    = 'Continuous'
    $script:bar.Minimum  = 0
    $script:bar.Maximum  = 100
    $script:bar.Value    = 3
    $script:bar.Size     = New-Object System.Drawing.Size(($W-40), 16)
    $script:bar.Location = New-Object System.Drawing.Point(20, 70)
    $script:form.Controls.Add($script:bar)

    # 닫기(X) = 시작 취소(서버도 종료)
    $btnClose = New-Object System.Windows.Forms.Label
    $btnClose.Text      = [char]0x2715
    $btnClose.Font      = $fontBody
    $btnClose.ForeColor = [System.Drawing.Color]::FromArgb(150, 150, 160)
    $btnClose.AutoSize  = $false
    $btnClose.TextAlign = 'MiddleCenter'
    $btnClose.Size      = New-Object System.Drawing.Size(22, 22)
    $btnClose.Location  = New-Object System.Drawing.Point(($W-28), 8)
    $btnClose.BackColor = [System.Drawing.Color]::Transparent
    $btnClose.Add_Click({ $script:cancelled = $true; $script:form.Close() })
    $script:form.Controls.Add($btnClose)

    $script:sw   = [System.Diagnostics.Stopwatch]::StartNew()
    $script:tick = 0

    $script:timer = New-Object System.Windows.Forms.Timer
    $script:timer.Interval = 250
    $script:timer.Add_Tick({
        $script:tick++
        $sec = [int]$script:sw.Elapsed.TotalSeconds
        $script:elapsed.Text = "$sec" + "초"

        if (-not $script:ready) {
            $pct = [int][Math]::Min(95, ($sec / $script:expectedSec) * 95)
            if ($pct -gt $script:bar.Value) { $script:bar.Value = $pct }
            if ($sec -ge $script:softMsgSec) {
                $script:status.Text = "조금 더 걸리고 있어요. 잠시만 기다려 주세요..."
            }
        }

        if ((($script:tick % 6) -eq 0) -and (-not $script:ready)) {
            if (Test-Ready) {
                $script:ready = $true
                $script:bar.Value   = 100
                $script:title.Text  = [char]0x2713 + "  준비 완료!"
                $script:status.Text = "앱 창을 엽니다..."
                $script:form.Refresh()
                $script:closeTimer = New-Object System.Windows.Forms.Timer
                $script:closeTimer.Interval = 900
                $script:closeTimer.Add_Tick({ $script:closeTimer.Stop(); $script:form.Close() })
                $script:closeTimer.Start()
            }
        }
    })
    $script:timer.Start()

    [void]$script:form.ShowDialog()
    $script:timer.Stop()
}

# =====================================================================
#  로딩바 종료 후 처리
# =====================================================================
if ($script:cancelled) {
    Stop-Server
    return
}

if (-not $script:ready) { return }

# --- 준비 완료: 전용 앱 창 열기 ---------------------------------------
if ($browser) {
    Remove-Item $appProfile -Recurse -Force -ErrorAction SilentlyContinue
    $appProc = Start-Process -FilePath $browser -ArgumentList @(
        "--app=$($script:url)",
        "--user-data-dir=$appProfile",
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1180,860"
    ) -PassThru

    if ($appProc) {
        # --- watchdog: 앱 창이 닫힐 때까지 대기 후 서버 종료 ---
        try { $appProc.WaitForExit() } catch { }
        Stop-Server
        Remove-Item $appProfile -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        Start-Process $script:url
    }
}
else {
    Start-Process $script:url
}
