# 获取所有包含 _temp.pt 的文件
$tempFiles = Get-ChildItem -Path ./checkpoints -Filter "*_temp.pt" -Recurse

if ($tempFiles.Count -eq 0) {
    Write-Host ">>> No temporary checkpoint files (*_temp.pt) found." -ForegroundColor Cyan
} else {
    Write-Host ">>> Found $($tempFiles.Count) temporary checkpoint files." -ForegroundColor Yellow
    foreach ($file in $tempFiles) {
        Write-Host "    Removing: $($file.FullName)"
        Remove-Item -Path $file.FullName -Force
    }
    Write-Host ">>> Cleanup complete." -ForegroundColor Green
}
