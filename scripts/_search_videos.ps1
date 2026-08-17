$shell = New-Object -ComObject Shell.Application
$pc = $shell.NameSpace(0x11)
$quest = $pc.Items() | Where-Object { $_.Name -eq 'Quest 3' }
$questFolder = $quest.GetFolder
$internal = $questFolder.Items().Item(0)

function Search-Folder($folder, $depth) {
    if ($depth -gt 4) { return }
    $items = $folder.GetFolder.Items()
    foreach ($item in $items) {
        $ext = [System.IO.Path]::GetExtension($item.Name).ToLower()
        if ($ext -eq '.mp4' -or $ext -eq '.webm' -or $ext -eq '.mkv') {
            Write-Host "[VIDEO] $($item.Name)  Size=$($item.Size)  Date=$($item.ModifyDate)"
        }
        if ($item.IsFolder) {
            Search-Folder $item ($depth + 1)
        }
    }
}

Write-Host "Searching all Quest 3 folders for videos..."
Search-Folder $internal 0
