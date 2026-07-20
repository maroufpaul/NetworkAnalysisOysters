foreach ($n in @(10,12,14)) {
  $seeds = @{10=50; 12=30; 14=15}[$n]
  foreach ($p0 in "constant","realistic") {
    .\.venv\Scripts\python.exe -m scripts.calibrate_real --n $n --seeds $seeds --matrix 2 --p0 $p0
  }
}
Write-Output "M2 DONE $(Get-Date)"
