$sizes = @(@(10,50), @(12,30), @(14,15))
foreach ($m in 1,2) {
  foreach ($p0 in "constant","realistic") {
    foreach ($s in $sizes) {
      python -m scripts.calibrate_real --n $s[0] --seeds $s[1] --matrix $m --p0 $p0
    }
  }
}
Write-Output "ALL DONE $(Get-Date)"
