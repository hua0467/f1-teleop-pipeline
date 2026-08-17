$shell = New-Object -ComObject Shell.Application

# Direct path to Quest 3 VideoShots via shell namespace
# Quest 3 -> Internal shared storage -> Oculus -> VideoShots
$questPath = "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}\\\?\usb#vid_2833&pid_5012&mi_00#6&2318dac3&0&0000#{6ac27878-a6fa-4155-ba85-f98f491d4f33}\SID-{10001,,481722765312}\Oculus\VideoShots"

$folder = $shell.NameSpace($questPath)
if ($folder -eq $null) {
    Write-Host "Cannot access VideoShots directly, trying step by step..."

    $pc = $shell.NameSpace(0x11)
    $quest = $pc.Items() | Where-Object { $_.Name -eq 'Quest 3' }
    if (-not $quest) { Write-Host "Quest 3 not found"; exit 1 }

    $storage = $quest.GetFolder.Items()
    Write-Host "Storage items:"
    foreach ($s in $storage) { Write-Host "  $($s.Name)" }

    # Navigate: Internal shared storage -> Oculus -> VideoShots
    $internal = $storage | Where-Object { $_.Name -match '内部|Internal' }
    if ($internal) {
        $oculus = $internal.GetFolder.Items() | Where-Object { $_.Name -eq 'Oculus' }
        if ($oculus) {
            Write-Host "Oculus items:"
            foreach ($o in $oculus.GetFolder.Items()) { Write-Host "  $($o.Name)" }
            $shots = $oculus.GetFolder.Items() | Where-Object { $_.Name -eq 'VideoShots' }
            if ($shots) {
                $videos = $shots.GetFolder.Items()
                Write-Host "`nVideoShots ($($videos.Count) files):"
                foreach ($v in $videos) {
                    Write-Host "  $($v.Name)  $($v.Size) bytes  $($v.ModifyDate)"
                }

                # Copy to PC
                $dest = "c:\Users\Administrator\Documents\f1-teleop-pipeline\videos"
                New-Item -ItemType Directory -Force -Path $dest | Out-Null
                foreach ($v in $videos) {
                    $srcPath = $v.Path
                    Write-Host "Copying $($v.Name) -> $dest\"
                    Copy-Item -Path $srcPath -Destination "$dest\$($v.Name)" -Force
                    Write-Host "  Done: $dest\$($v.Name)"
                }
            }
        }
    }
} else {
    Write-Host "Found VideoShots directly"
    $videos = $folder.Items()
    foreach ($v in $videos) {
        Write-Host "  $($v.Name)  $($v.Size) bytes"
    }
}
