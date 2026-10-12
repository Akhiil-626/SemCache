function Ask($model, $text, $noCache = $false) {
  $body = @{ model = $model; messages = @(@{ role = "user"; content = $text }) } | ConvertTo-Json -Depth 5
  $headers = @{}
  if ($noCache) { $headers["Cache-Control"] = "no-cache" }
  try {
    Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType "application/json" -Headers $headers -Body $body
  } catch {
    $sr = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
    $sr.ReadToEnd()
  }
}
