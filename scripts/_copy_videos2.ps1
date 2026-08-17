$shell = New-Object -ComObject Shell.Application
$pc = $shell.NameSpace(0x11)
$quest = $pc.Items() | Where-Object { $_.Name -eq 'Quest 3' }

# Navigate into Quest 3
$questFolder = $quest.GetFolder
$items = $questFolder.Items()
Write-Host "=== Quest 3 root ($($items.Count) items) ==="
foreach ($item in $items) {
    Write-Host "Name: $($item.Name)"
    Write-Host "IsFolder: $($item.IsFolder)"
    Write-Host "Path: $($item.Path)"
    Write-Host "---"
}

# Try to get the first item (should be Internal shared storage)
if ($items.Count -gt 0) {
    $storage = $items.Item(0)
    Write-Host "Storage selected: $($storage.Name)"
    $storageFolder = $storage.GetFolder
    $storageItems = $storageFolder.Items()
    Write-Host "Storage items ($($storageItems.Count)):"
    foreach ($si in $storageItems) {
        Write-Host "  $($si.Name) (IsFolder=$($si.IsFolder))"
        if ($si.Name -eq 'Oculus') {
            Write-Host "  -> Found Oculus!"
            $ocFolder = $si.GetFolder
            $ocItems = $ocFolder.Items()
            foreach ($oc in $ocItems) {
                Write-Host "    $($oc.Name)"
                if ($oc.Name -eq 'VideoShots') {
                    Write-Host "    -> Found VideoShots!"
                    $vsFolder = $oc.GetFolder
                    $vsItems = $vsFolder.Items()
                    Write-Host "    Videos ($($vsItems.Count)):"

                    $dest = "c:\Users\Administrator\Documents\f1-teleop-pipeline\videos"
                    New-Item -ItemType Directory -Force -Path $dest | Out-Null

                    foreach ($vid in $vsItems) {
                        Write-Host "      $($vid.Name) $($vid.Size) bytes"
                        # Copy using shell copy
                        $destFolder = $shell.NameSpace($dest)
                        if ($destFolder) {
                            $destFolder.CopyHere($vid, 16)  # 16 = respond with "yes to all"
                            Write-Host "      -> Copied!"
                        }
                    }
                }
            }
        }
    }
}
