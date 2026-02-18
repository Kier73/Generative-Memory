$releaseDir = "release"
$zipName = "generative_memory_v0.1.0_win64_avx2.zip"

if (Test-Path $zipName) {
    Remove-Item $zipName
}

Compress-Archive -Path "$releaseDir\*" -DestinationPath $zipName -Force
Write-Host "Created $zipName"
