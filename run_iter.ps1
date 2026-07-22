foreach ($m in 1,2) {
  foreach ($nk in @(@(10,50),@(12,30),@(14,15))) {
    .\.venv\Scripts\python.exe -m scripts.calibrate_iterated --n $nk[0] --seeds $nk[1] --matrix $m
  }
}
Write-Output "ITER DONE $(Get-Date)"
