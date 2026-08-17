$shell = New-Object -ComObject Shell.Application
$pc = $shell.NameSpace(0x11)
$quest = $pc.Items() | Where-Object { $_.Name -eq 'Quest 3' }
$questFolder = $quest.GetFolder
$internal = $questFolder.Items().Item(0)
$oculus = $internal.GetFolder.Items() | Where-Object { $_.Name -eq 'Oculus' }
$shots = $oculus.GetFolder.Items() | Where-Object { $_.Name -eq 'VideoShots' }
$videos = $shots.GetFolder.Items()

Write-Host "=== VideoShots ($($videos.Count) files) ==="
$dest = "c:\Users\Administrator\Documents\f1-teleop-pipeline\videos"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$destFolder = $shell.NameSpace($dest)

foreach ($vid in $videos) {
    Write-Host "$($vid.Name)  $([math]::Round($vid.Size/1MB, 2)) MB  $($vid.ModifyDate)"
    # Only copy new files
    $targetPath = Join-Path $dest $vid.Name
    if (-not (Test-Path $targetPath)) {
        $destFolder.CopyHere($vid, 16)
        Write-Host "  -> Copied"
    } else {
        Write-Host "  -> Already exists"
    }
}
