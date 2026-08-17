$shell = New-Object -ComObject Shell.Application
$pc = $shell.NameSpace(0x11)
Write-Host "=== Devices in This PC ==="
foreach ($item in $pc.Items()) {
    Write-Host "  $($item.Name)"
}
