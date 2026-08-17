$shell = New-Object -ComObject Shell.Application
$pc = $shell.NameSpace(0x11)
$quest = $pc.Items() | Where-Object { $_.Name -eq 'Quest 3' }
if ($quest) {
    Write-Host "Quest 3 found, navigating..."
    # Navigate: Quest 3 -> Internal shared storage -> Oculus -> VideoShots
    $storage = $quest.GetFolder.Items()
    Write-Host "Root items:"
    foreach ($item in $storage) {
        Write-Host "  $($item.Name) ($($item.Path))"
    }
    $internal = $quest.GetFolder.Items() | Where-Object { $_.Name -match 'Internal|内部' }
    if ($internal) {
        Write-Host "`nInternal storage items:"
        foreach ($item in $internal.GetFolder.Items()) {
            Write-Host "  $($item.Name)"
            if ($item.Name -eq 'Oculus') {
                Write-Host "`nOculus items:"
                foreach ($oc in $item.GetFolder.Items()) {
                    Write-Host "  $($oc.Name)"
                    if ($oc.Name -eq 'VideoShots') {
                        Write-Host "`nVideoShots files:"
                        foreach ($vid in $oc.GetFolder.Items()) {
                            Write-Host "  $($vid.Name)  ($($vid.Size) bytes)  $($vid.ModifyDate)"
                        }
                    }
                }
            }
        }
    }
}
